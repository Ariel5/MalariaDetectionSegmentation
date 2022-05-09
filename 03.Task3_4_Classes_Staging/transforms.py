import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from dlt.common.transforms import PILLoader, BaseTransform, Binarize
from network_functions import OnlyKeep, USResize, USToTensor, FlattenStudy
from copy import copy
import torchvision
import random
import datetime
import torch
import numpy

class Transforms_Augmentation(BaseTransform):

    def __init__(self, fields, transforms):
        super().__init__(fields)
        self.transforms = torchvision.transforms.Compose(transforms)

    def __call__(self, data_dict):
        super().__call__(data_dict)

        random.seed(datetime.datetime.now())
        seed = numpy.random.randint(2147483647)

        for field in self.fields:
            random.seed(seed)
            data_dict[field] = self.transforms(data_dict[field])

        return data_dict

class ThresholdToClass(BaseTransform):
    """
    Binarize numpy arrays, or torch.tensors. Note, if doing it to
    torch.Tensor, this will copy to cpu and perform numpy operation
    """
    def __init__(self, fields, threshold=0.5, new_min=0, new_max=1, dtype=numpy.float32):
        """
        threshold: threshold value
        new_min: new value for values below threshold
        new_max: new value for values greater or equal to threshold
        dtype: dtype of resulting array
        """
        super().__init__(fields)
        self._threshold = threshold
        self._new_min = new_min
        self._new_max = new_max
        self.dtype = dtype

    def __call__(self, data_dict):
        for field in self.fields:
            val = data_dict[field]
            is_torch = False
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
                is_torch = True

            result = len(self._threshold)
            for i in range(len(self._threshold)):
                if val<self._threshold[i]:
                    result = i
                    break
            
            data_dict[field] = numpy.array(result)
# 3.4])
            # data_dict[field] = np.where(val >= self._threshold[0],
                                        # self._new_max, self._new_min).astype(self.dtype)
            # print (val, data_dict[field], type(data_dict[field]))
            # return None

            # convert back to tensor if needed
            if is_torch:
                data_dict[field] = torch.from_numpy(data_dict[field])
        return data_dict



class ThresholdToClass_FrankHall(BaseTransform):
    def __init__(self, fields, threshold):
        super().__init__(fields)
        self._threshold = threshold

    def __call__(self, data_dict):
        for field in self.fields:
            val = data_dict[field]
            if val in [0,1]:
                data_dict[field] = numpy.array([0,0]).astype(numpy.float32)
            elif val in [2]:
                data_dict[field] = numpy.array([1,0]).astype(numpy.float32)
            elif val in [3]:
                data_dict[field] = numpy.array([1,1]).astype(numpy.float32)
            print (val)
            print (data_dict[field])
            data_dict[field] = torch.from_numpy(data_dict[field])
            # is_torch = False
            # if isinstance(val, torch.Tensor):
            #     val = val.cpu().numpy()
            #     is_torch = True

            # result = [0.0 for _ in range(len(self._threshold))]
            # for i in range(len(self._threshold)):
            #     if val>self._threshold[i]:
            #         result[i] = 1.0
            
            # data_dict[field] = numpy.array(result).astype(numpy.float32)

            # if is_torch:
            #     data_dict[field] = torch.from_numpy(data_dict[field])
        return data_dict

class To_Long(BaseTransform):

    def __init__(self, fields):
        super().__init__(fields)
        
    def __call__(self, data_dict):
        super().__call__(data_dict)

        for field in self.fields:
            data_dict[field] = data_dict[field].astype(numpy.int)
        return data_dict

class To_Float(BaseTransform):

    def __init__(self, fields):
        super().__init__(fields)
        
    def __call__(self, data_dict):
        super().__call__(data_dict)

        for field in self.fields:
            data_dict[field] = data_dict[field].astype(numpy.float32)
        return data_dict




def Transforms_Study(im_root, new_size, im_fields=['im'], label_field=['steatosis'], extra_keeps=[], image_augmention=None, label_binarize_threshold=None):

    transforms = []
    keep_fields = copy(im_fields)
    keep_fields.extend(label_field)
    keep_fields.extend(extra_keeps)

    transforms.append(OnlyKeep(fields=keep_fields))
    transforms.append(PILLoader(fields=im_fields, root_dir=im_root, mode='RGB'))
    transforms.append(USResize(fields=im_fields, new_size=new_size))

    if image_augmention:
        transforms.append(Transforms_Augmentation(fields=im_fields, transforms=image_augmention))

    if label_binarize_threshold:
        print ('here')
        if type(label_binarize_threshold) is not list:
            transforms.append(Binarize(label_field, label_binarize_threshold))
            transforms.append(To_Long(label_field))
        else:
            transforms.append(ThresholdToClass(label_field, label_binarize_threshold))
            transforms.append(To_Long(label_field))

    transforms.append(USToTensor(fields=im_fields))

    transforms = torchvision.transforms.Compose(transforms)
    return transforms





# def Transforms_Study_FrankHall(im_root, new_size, im_fields=['im'], label_field=['steatosis'], extra_keeps=[], image_augmention=None, label_binarize_threshold=None):

#     transforms = []
#     keep_fields = copy(im_fields)
#     keep_fields.extend(label_field)
#     keep_fields.extend(extra_keeps)

#     transforms.append(OnlyKeep(fields=keep_fields))
#     transforms.append(PILLoader(fields=im_fields, root_dir=im_root, mode='RGB'))
#     transforms.append(USResize(fields=im_fields, new_size=new_size))

#     if image_augmention:
#         transforms.append(Transforms_Augmentation(fields=im_fields, transforms=image_augmention))

#     transforms.append(ThresholdToClass_FrankHall(label_field, label_binarize_threshold))

#     transforms.append(USToTensor(fields=im_fields))

#     transforms = torchvision.transforms.Compose(transforms)
#     return transforms









# def Transforms_Study_FrankHall(im_root, new_size, im_fields=['im'], label_field=['steatosis'], extra_keeps=[], image_augmention=None):

#     transforms = []
#     keep_fields = copy(im_fields)
#     keep_fields.extend(label_field)
#     keep_fields.extend(extra_keeps)

#     transforms.append(OnlyKeep(fields=keep_fields))
#     transforms.append(PILLoader(fields=im_fields, root_dir=im_root, mode='RGB'))
#     transforms.append(USResize(fields=im_fields, new_size=new_size))

#     if image_augmention:
#         transforms.append(Transforms_Augmentation(fields=im_fields, transforms=image_augmention))

#     # transforms.append(ThresholdToClass_FrankHall(label_field, label_binarize_threshold))

#     transforms.append(USToTensor(fields=im_fields))

#     transforms = torchvision.transforms.Compose(transforms)
#     return transforms





