import torch
import random
import numpy as np
from copy import deepcopy
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset
from dlt.common.monitors import ValuesCollectorNew
from dlt.common.core import Component
from dlt.common.datasets import BaseDataset


class StudyMedianMetric(ValuesCollectorNew):
    """
    Generic class for performing evaluation metrics by taking the median score 
    for each study. 
    Assumes there are image-wise scores, which are aggregated together by 
    grouping them based on the patient_id and scanner_type
    """

    def __init__(self, pred_tag, label_tag, scanner_tag, id_tag, output_var,
                 do_train=True, do_val=True):

        """
        Args:

        pred_tag: real-valued context variable holding the ranking scores
        label_tag: the true class indices
        scanner_tag: context var for the scanner name 
        id_tag: context var for the patient id
        output_var: context var to score the metric. An inheriting class must
            implement this functionality
        """

        # collect the preds, labels, scanners and patient_ids
        super().__init__([pred_tag, label_tag,
                          scanner_tag, id_tag], do_train, do_val)
        self.pred_tag = pred_tag
        self.label_tag = label_tag
        self.scanner_tag = scanner_tag
        self.id_tag = id_tag
        self.output_tag = output_var

    def callback_post_train_epoch(self, context):
        if self.do_train:
            self._calculate(context)

    def callback_post_val_epoch(self, context):
        if self.do_val:
            self._calculate(context)

    # execute study_wise Median AUC
    def _calculate(self, context):
        """
        Calculate the median score across each id+scanner tag
        """
        from collections import defaultdict
        from statistics import median

        study_dict = defaultdict(list)
        label_dict = {}

        try:
            _ = context[self.scanner_tag]
            self.value_dict = context
            # print ('it is called')
        except:
            pass

        scanners = self.value_dict[self.scanner_tag]
        ids = self.value_dict[self.id_tag]
        outputs = self.value_dict[self.pred_tag]
        labels = self.value_dict[self.label_tag]
        
        # print (len(scanners))
        # print (len(ids))
        # print (len(outputs))
        # print (len(labels))


        # create a dict that stores a list of score values for each id+scanner
        for (cur_id, scanner, prediction, label) in zip(ids, scanners, outputs, labels):
            study_dict[cur_id + scanner].append(float(prediction))
            # this should the same value each time
            label_dict[cur_id + scanner] = int(label)
            
        # print ('herer')
        # # print (study_dict)
        # results = []
        # import pandas
        # for k, v in study_dict.items():
        #     results.append([k,v])
        # df = pandas.DataFrame(results)
        # df.to_csv('123.csv')

        # now create a list of median pred scores plus the label
        pred_list = []
        label_list = []
        for k, v in study_dict.items():
            pred_list.append(median(v))
            label_list.append(label_dict[k])

        label_list = np.asarray(label_list)
        pred_list = np.asarray(pred_list)

        # execute the metric
        self._do_metric(context, label_list, pred_list)

    def _do_metric(self, context, label_list, pred_list):

        raise Error('Not implemented')


class StudyWiseMedianAUC(StudyMedianMetric):
    """
    Class for compute study-wise AUC by taking median across each study
    """
    def __init__(self, pred_tag, label_tag, scanner_tag, id_tag, output_var,
                 do_train=True, do_val=True):

        """
        Args:
            pred_tag: real-valued context variable holding the ranking scores
            label_tag: the true class indices
            scanner_tag: context var for the scanner name 
            id_tag: context var for the patient id
            output_var: context var to score the AUC
        """
        super().__init__(pred_tag, label_tag, scanner_tag,
                         id_tag, output_var, do_train, do_val)

    def _do_metric(self, context, label_list, pred_list):

        auc = roc_auc_score(label_list, pred_list)
        context[self.output_tag] = auc


class StudyWiseJonckheereTerpstra(StudyMedianMetric):
    """
    Class for measuring multi-partite ranking performance, see eqn (3) in 
    "Binary Decomposition Methods for Multipartite Ranking"
    Assumes a ranking score is available in the context, e.g., after
    MultiClass2Score is executed

    This class takes the median score across each study
    """
    def __init__(self, pred_tag, label_tag, scanner_tag, id_tag, output_var, num_classes,
                 do_train=True, do_val=True):
        """
        Args:
            pred_tag: real-valued context variable holding the ranking scores
            label_tag: the true class indices
            scanner_tag: context var for the scanner name 
            id_tag: context var for the patient id
            output_var: context var to score the JH score
            num_classes: the number of class indices
        """

        super().__init__(pred_tag, label_tag, scanner_tag,
                         id_tag, output_var, do_train, do_val)
        self.num_classes = num_classes

    # execute study_wise Median AUC
    def _do_metric(self, context, label_list, pred_list):

        from sklearn.metrics import roc_auc_score

        auc_sum = 0
        combo_count = 0

        # for each pair of indices, i,j, where i < j: compute the AUC
        for i_class in range(self.num_classes-1):
            # extract preds and labels for class i
            cur_i_idx = np.where(label_list == i_class)[0]
            if len(cur_i_idx) == 0:
                continue
            cur_i_labels = label_list[cur_i_idx]
            cur_i_preds = pred_list[cur_i_idx]
            for j_class in range(i_class+1, self.num_classes):
                # extract preds and labels for class j
                cur_j_idx = np.where(label_list == j_class)[0]
                if len(cur_j_idx) == 0:
                    continue
                cur_j_labels = label_list[cur_j_idx]
                cur_j_preds = pred_list[cur_j_idx]
                cur_labels = np.concatenate([cur_i_labels, cur_j_labels])
                cur_preds = np.concatenate([cur_i_preds, cur_j_preds])
                auc_sum += roc_auc_score(cur_labels, cur_preds)
                combo_count += 1
        # normalized the score to be in [0,1]
        auc_sum *= 2/(self.num_classes*(self.num_classes-1))

        context[self.output_tag] = auc_sum

        # print ('labels')
        # print (label_list)
        # print ('preds')
        # print (pred_list)
        # print ('hererere')


class MultiClass2Score(Component):
    """
    class for turning different styles of prediction results into a ranking
    score
    """

    def __init__(self, pred_tag, output_tag, class_indices, category):
        """
        Args:
            pred_tag: context variable name for the model outputs
            output_tag: context variable name to store the resulting ranking
            class_indices: index for each class, e.g., [0,1,2] for 3 classes
                (only used for multi_class) 
            category: type of predictions, can be multi-class or frank-hall 
                (cumulative probability scores)
        """
        super().__init__()

        self.pred_tag = pred_tag
        self.output_tag = output_tag
        self.class_indices = class_indices
        self.category = category
        if self.category == 'multi_class':
            # unsqueeze the class indices so we can broadcast them
            if len(self.class_indices.shape) == 1:
                self.class_indices = torch.unsqueeze(self.class_indices, 0)


    def _create_score_frankhall(self, context):
        """
        if predictions are cumlative probabilities, the ranking score is 
        simply the sum of each probability after a sigmoid. See eqn (7) in
        "Binary Decomposition Methods for Multipartite Ranking"
        """
        cur_preds = context[self.pred_tag]
        cur_preds = torch.sigmoid(cur_preds)
        score = torch.sum(cur_preds, 1)
        context[self.output_tag] = score

    def _create_score_multiclass(self, context):
        """
        If predictions are standard multi-class, then the ranking score is
        simply the sum of each class pseudo-prob weighted by the class index.
        See eqn (7) and the following equation in "Binary Decomposition 
        Methods for Multipartite Ranking"
        """

        cur_preds = context[self.pred_tag]
        cur_preds = torch.nn.functional.softmax(cur_preds, 1)
        cur_preds *= self.class_indices
        score = torch.sum(cur_preds, 1)
        context[self.output_tag] = score

    def _create_score(self, context):
        if self.category in ['multi_class', 'vanilla']:
            self._create_score_multiclass(context)
        elif self.category == 'frank_hall':
            self._create_score_frankhall(context)

    def callback_post_val_iter(self, context):
        self._create_score(context)

    def callback_post_train_iter(self, context):
        self._create_score(context)



class StudyDataset(Dataset):
    """
    Dataset to load images within a study. Nested dataset structure, each study has its own
    image dataset. Restricted (due to pytorch's expectations in collate_fn) to always return
    the same number of images per study. Optional invalid can be provided to fill in dummy
    items if the study does not have enough images
    """

    def __init__(self, listdict, num_to_return, study_transforms=None, im_transforms=None, im_list_field='im_list',
            invalid_callable=None):
        """
        Args:

        listdict: list of dictionaries, each entry represents a study
        num_to_return: number of images dictionaries to return upon each call. 
          If the study has more images, a random sample will be collected
        study_transform: optional transform to apply to each study_dict after
          im_dicts are loaded. Common choice is FlattenStudy
        im_transforms: transforms to apply to each entry in the im_dicts being
          loaded. Same transforms you would be using for a standard BaseDataset
        im_list_field: the field in the study dict that holds the list of image
          dicts
        invalid_callable: optional callable object to be used to fill in im_dict
          entry if the study does not have enough images. Should be filled in with
          values, e.g., -1, that make it obvious for downsteam processing to ignore.
          this is needed, because pytorch collate_fn assumes the same size for 
          every entry when it makes a batch


        """

        self._listdict = listdict
        self.im_list_field = im_list_field
        self.num_to_return = num_to_return
        self.study_transforms = study_transforms
        self.invalid_callable = invalid_callable
        # we create a seperate list of datasets, because we don't want them copied
        # during __getitem__
        # create a dataset for each list of im_dicts in each study, making sure we 
        # pass the im_transforms
        self._dataset_list = [BaseDataset(cur_dict[self.im_list_field], transforms=im_transforms)
                for cur_dict in self._listdict]

    def __len__(self):
        return len(self._listdict)


    def __getitem__(self, idx):
        # get the current study dict, do a deepcopy so we don't override
        # any fields from the self._listdict
        cur_dict = deepcopy(self._listdict[idx])
        # get the current im_dict dataset for this study
        cur_dataset = self._dataset_list[idx]
        num_images = len(cur_dataset)
        # if the study has more images than needed, we randomly sample them
        if num_images  > self.num_to_return:
            im_indices = random.sample(range(num_images), k=self.num_to_return)
        else:
            im_indices = range(num_images)
        # get each corresponding im_dict. Individual datasets will perform
        # any necessary transforms (e.g., loading)
        im_list = [cur_dataset[im_idx] for im_idx in im_indices]

        # if the study does not have enough images, fill in the remainder with
        # dummy invalid entries
        if len(im_list) < self.num_to_return:
            dummy_list = [self.invalid_callable() for i in range(self.num_to_return - len(im_list))]
            im_list.extend(dummy_list)

        # now set the loaded list of im_dicts to the appropriate field
        cur_dict[self.im_list_field] = im_list
        # if there are any study transforms, execute them before returning
        if self.study_transforms:
            cur_dict = self.study_transforms(cur_dict)


        return cur_dict

       
class ModelEvalSetter(Component):

    def __init__(self, model_var_name):
        super().__init__()
        self._model_var_name = model_var_name

    def callback_pre_train_iter(self, context):
        context[self._model_var_name].eval()
        



































