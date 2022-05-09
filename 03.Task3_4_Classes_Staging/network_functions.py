import torch
from copy import copy
import numpy as np
from torch.utils.data import DataLoader
import os
import torchvision
from tqdm import tqdm
import torch.utils.tensorboard
import functions_supporting
import sklearn
import numpy
from dlt.common.datasets import BaseDataset
from dlt.common.transforms import PILLoader, BaseTransform
import PIL


def wce_wrapper(weight):
    weights = torch.FloatTensor(weight).cuda()
    return torch.nn.CrossEntropyLoss(weight=weights)

def bce_wrapper(pos_weight=None):
    if pos_weight:
        pos_weight = torch.FloatTensor(pos_weight).cuda()
    return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)



class ReverseCE(torch.nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.weights = weights
        if self.weights is not None:
            self.weights = torch.Tensor(self.weights).cuda()
    def forward(self, input, target):
        if self.weights is not None:
            cur_weights = self.weights[target.long()]
        target = target.unsqueeze(1)
        target_one_hot = torch.zeros_like(input).to(input.device)
        target_one_hot.scatter_(1, target, value=1)
        input = self.logsoftmax(input)

        loss = target_one_hot * input - input
        # sum across target indices first so we can weight properly
        loss = torch.sum(loss, dim = 1)
        if self.weights is not None:
            loss = loss * cur_weights
        loss = torch.sum(loss)/((input.shape[1]-1)* input.shape[0])
        return loss

class SoftLabels(torch.nn.Module):
    """
    Simple wrapper to make sure the input is squeeze to a vector
    """
    def __init__(self, pos_weight=None):
        super().__init__()
        # make sure the weight is in cuda
        if pos_weight:
            pos_weight = torch.FloatTensor(pos_weight).cuda()
        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
    def forward(self, input, target):

        input = torch.nn.functional.log_softmax(input, 1)
        loss = input * target
        loss = -1 * torch.mean(loss)

        return loss

class FrankHalklLSE(torch.nn.Module):
    """
    Simple wrapper to make sure the input is squeeze to a vector
    """
    def __init__(self):
        super().__init__()
        # make sure the weight is in cuda
        self.loss = torch.nn.BCELoss()
    def forward(self, input, target):

        cum_probs = []
        cum_probs_neg = []
        total_prob = torch.logsumexp(input, 1)

        for i in range(1, input.shape[1]):
            cum_probs.append(torch.logsumexp(input[:,i:], 1) - total_prob)
            cum_probs_neg.append(torch.logsumexp(input[:,:1], 1) - total_prob)

        cum_probs = torch.stack(cum_probs, dim=1)
        cum_probs_neg = torch.stack(cum_probs_neg, dim=1)

        loss = target * cum_probs + (1-target) * cum_probs_neg

        return torch.mean(-1 * loss)





criteria = {'CE': torch.nn.CrossEntropyLoss,
            'MSE': torch.nn.MSELoss,
            'L1': torch.nn.L1Loss,
            'WCE': wce_wrapper,
            'RCE': ReverseCE,
            'BCE': bce_wrapper,
            'SoftLabels': SoftLabels,
            }



class USToTensor(BaseTransform):
    def __init__(self, fields):
        super().__init__(fields)

    def __call__(self, data_dict):
        super().__call__(data_dict)
        for field in self.fields:
            data_dict[field] = torchvision.transforms.functional.to_tensor(data_dict[field])

        return data_dict

class OnlyKeep(BaseTransform):
    def __init__(self, fields):
        super().__init__(fields)

    def __call__(self, data_dict):
        super().__call__(data_dict)
        new_data_dict = {}
        for field in self.fields:
            new_data_dict[field] = data_dict[field]

        return new_data_dict

class USAugment(BaseTransform):

    def __init__(self, fields):
        super().__init__(fields)
        self.transforms  = [
            torchvision.transforms.ColorJitter(brightness=.1, contrast=.1),
            torchvision.transforms.RandomAffine(degrees=10, scale=(.9,1.1),
                                     shear=2, resample=PIL.Image.BILINEAR)]
        self.transforms = torchvision.transforms.Compose(self.transforms)

    def __call__(self, data_dict):
        super().__call__(data_dict)

        for field in self.fields:
            data_dict[field] = self.transforms(data_dict[field])

        return data_dict

class USResize(BaseTransform):

    def __init__(self, fields, new_size):
        super().__init__(fields)
        self.new_size = new_size
    def __call__(self, data_dict):
        super().__call__(data_dict)

        for field in self.fields:
            data_dict[field] = self.get_resized_image(data_dict[field])

        return data_dict

    def get_h_w_resize_keep_ratio(self, h, w):
        if h > w:
            return self.new_size, int(self.new_size*w/h)
        return int(self.new_size*h/w), self.new_size

    def get_padding_parameters(self, h, w):
        patch_size = self.new_size
        if w == patch_size:
            return (patch_size-h)//2, 0
        return 0, (patch_size-w)//2

    def get_resized_image(self, image):
        w, h = image.size
        h, w = self.get_h_w_resize_keep_ratio(h, w)
        ph, pw = self.get_padding_parameters(h, w)
        image = torchvision.transforms.functional.resize(image, (h, w))
        image = torchvision.transforms.functional.pad(image, (pw, ph))
        image = torchvision.transforms.functional.resize(
            image, (self.new_size, self.new_size))

        return image


class FlattenStudy(BaseTransform):
    """
    A study will have a list of dicts for each image, these should be flattened
    together prior to any batching by the DataLoader. This allows for natural
    use within pytorch
    """
    def __init__(self, im_dict_field):
        super().__init__(im_dict_field)
        self._im_dict_field = im_dict_field
        
    def __call__(self, study_dict):

        # get the list of im_dicts from the study_dict
        im_list_dict = copy(study_dict[self._im_dict_field])
        # get the first im_dict
        first_elem = im_list_dict[0]

        # for key, value in first_elem.items():
        #     print (key)
        # print (first_elem.items())

        # now for each entry in the im_dicts, we will yank them out of the 
        # list and stack them together into one array and then store in the
        # actual study_dict itself
        for key, value in first_elem.items():
            # create a list of every entry for the current entry
            val_list = [elem[key] for elem in im_list_dict if elem ]

            # depending on its type, we stack them different ways
            # we then take the key in the list and set as a key, val pair
            # in the parent study_dict
            if isinstance(value, torch.Tensor):
                study_dict[key] = torch.stack(val_list)
            elif isinstance(value, np.ndarray):
                study_dict[key] = np.stack(val_list)
            # we assume we can stack anything else. May require a fix later
            else:
                study_dict[key] = np.stack(val_list)
        # now we pop out the list of im_dicts, since we no longer need it, 
        # as each element is now stored directly in the study_dict itself
        # (after stacking)
        study_dict.pop(self._im_dict_field)
        return study_dict

def USTransformsTrain(im_root, new_size, im_field='im', label_field='view_group', extra_keeps=[]):



    transforms = []
    keep_fields = [im_field, label_field]
    keep_fields.extend(extra_keeps)
    transforms.append(OnlyKeep(fields=keep_fields))
    transforms.append(PILLoader(fields=im_field, root_dir=im_root, mode='RGB'))
    transforms.append(USResize(fields=im_field, new_size=new_size))
    transforms.append(USAugment(fields=im_field))
    transforms.append(USToTensor(fields=im_field))
    transforms = torchvision.transforms.Compose(transforms)
    return transforms

def USTransformsEval(im_root, new_size, im_field='im', label_field='view_group', extra_keeps=[]):



    transforms = []

    keep_fields = [im_field, label_field]
    keep_fields.extend(extra_keeps)
    transforms.append(OnlyKeep(fields=keep_fields))
    transforms.append(PILLoader(fields=im_field, root_dir=im_root, mode='RGB'))
    transforms.append(USResize(fields=im_field, new_size=new_size))
    transforms.append(USToTensor(fields=im_field))
    transforms = torchvision.transforms.Compose(transforms)
    return transforms




def check_trainable_parameter_number(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel()
                                 for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

