import sys
import warnings

###############################################################################################
###### Libraries: Common
###############################################################################################
import pandas
import os
import argparse
import torch
import torchvision
import PIL
import sys
import numpy
import copy
import json
sys.path.append('../')

###############################################################################################
###### Libraries: Project Specific
###############################################################################################
import functions_supporting
from network_functions import USTransformsTrain, USTransformsEval, criteria
import network
import dlt_components
import dlt_extension_metrics
import transforms
from dlt_components import MultiClass2Score, StudyWiseJonckheereTerpstra 
import functions_data_preprocessing

###############################################################################################
###### Libraries: Differentiable Learning Toolkit
###############################################################################################
import dlt
import dlt.common.utils
from dlt.common.transforms import MoveToCuda, Binarize, PreIterTransform, PostIterTransform, Callable
from dlt.common.utils.setup import read_yaml, init_workspace
from dlt.common.datasets import BaseDataset, RandomSamplerWithLength
from dlt.common.core import Context
from dlt.common.metrics import OnlineAverageLoss, ClassificationAccuracyNew

from dlt.common.monitors import PrintEpochLogger, TQDMConsoleLogger, TensorboardLogger
from dlt.common.trainers import SupervisedTrainer
from dlt.common.controllers import ModelSaver
from dlt.common.layers.batch_wrappers import MetricWrapper


def run_test(args):

    ################################################################################################################
    ##### Environment Setup
    ################################################################################################################
    torch.cuda.set_device(args.gpu)
    config = read_yaml(args.yaml_file)
    
    # Create context
    context = Context()
    context['config'] = config

    for fold in range(5):

        ################################################################################################################
        ##### Data Preparation
        ################################################################################################################
        data_validation = functions_data_preprocessing.read_json(args.datasheet_root + 'Dev_image_list_'+str(fold)+'.json')

        # Step 1.1 Define transforms
        transforms_validation = transforms.Transforms_Study(
            im_root = config.data.path,
            new_size = config.data.patch_size,
            im_fields = ['im'],
            label_field = [config.data.predict],
            image_augmention = None,
            )

        # Step 1.2 Construct datasets, dataloaders
        dataset_validation = BaseDataset(
            listdict = data_validation, 
            transforms = transforms_validation,
            )
        dataloader_validation = torch.utils.data.DataLoader(
            dataset_validation,
            batch_size = config.solver.batch_size, 
            shuffle = False,
            num_workers = config.solver.num_workers,
            pin_memory = True,
            )

        image_names = [item['im'] for item in data_validation]
        # it = iter(dataloader_validation)
        # first = next(it)
        # print (first)
        # print (len(dataloader_validation))
        

        ###############################################################################################
        ###### Step 2. Model Preparation
        ###############################################################################################

        # Step 2.0 Set up model
        model = network.WrappedNetwork(
            config = config,
            im_key = 'im',
            ).cuda()
        context['component.model'] = model
        context.load('./ckpts/'+config.study.name+'-F'+str(fold)+'/best_model.ckpt')
        model = context['component.model']._model
        model = model.cuda()
        model.eval()

        # class_indices = [i for i in range(len(config.data.thresholds))]
        # class_indices = torch.Tensor(class_indices).cuda()
        # score_creator = MultiClass2Score('var.batch_data.output', 'var.metric.score', class_indices, config.model.category)

        # jt_field = 'old_labels'

        # jt_calculator = StudyWiseJonckheereTerpstra(
        #     pred_tag = 'var.metric.score',
        #     label_tag = 'var.batch_data.'+jt_field,
        #     scanner_tag = 'var.batch_data.Study',
        #     id_tag = 'var.batch_data.Study',
        #     output_var = 'var.metric.jt',
        #     num_classes = len(config.data.thresholds)+1,
        # )

        list_yhat = []
        list_old_labels = []
        list_study = []

        result = []
        counter_name = 0
        with torch.no_grad():
            for i, batch in enumerate(dataloader_validation):
                x = batch['im'].cuda()

                y_hat = model.forward(x)

                for j in range(len(y_hat)):
                    result.append({
                        'image': image_names[counter_name].split('/')[-1],
                        'gt':batch['label'][j].item(),
                        'predicted':y_hat[j][0].item(),
                    })
                    counter_name+=1
        result = pandas.DataFrame(result)
        result.to_excel(args.save_path+'/results/prediction_dev_f'+str(fold)+'.xlsx', index=False)

            
# python train_bigdata_multiclass.py --im_root '/home/lbwdruid/Desktop/Liver_Ultrasound/0000.data/0006.bigdata/' --yaml_file  'hyper-parameters-bigdata.yml' --datasheet_root '/home/lbwdruid/Desktop/Liver_Ultrasound/0000.data/0006.bigdata/lists/entropy/' --save_path './' --fold 2 --gpu 3
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', type=str, help="path to yaml file for configuration", default='hyper-parameters.yml')
    parser.add_argument('--save_path', type=str, help="path to where you want checkpoints and logs save to", default='hyper-parameters.yml')
    parser.add_argument('--datasheet_root', type=str, help="path to folder with datasheets", default='./')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu ids to use')
    args = parser.parse_args()

    run_test(args)

































