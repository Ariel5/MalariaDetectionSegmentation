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

    fold_calculations = []
    import tqdm
    for fold in tqdm.tqdm([0,1,2,3,4]):
    # for fold in tqdm.tqdm([3,4]):
        
        ################################################################################################################
        ##### Data Preparation
        ################################################################################################################
        data_validation = functions_data_preprocessing.read_json(args.datasheet_root + 'Test_image_list.json')

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

        result = []
        counter_name = 0

        list_gt = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader_validation):
                x = batch['im'].cuda()

                y_hat = model.forward(x)

                for j in range(len(y_hat)):
                    prediction = torch.argmax(y_hat[j]).item()
                    result.append({
                        'image': image_names[counter_name].split('/')[-1],
                        'gt':batch['label'][j].item(),
                        'predicted':prediction,
                    })
                    counter_name+=1
        fold_calculations.append(result)
    
    result = []
    list_prediction = []
    list_gt = []
    for i in range(len(fold_calculations[0])):
        image = fold_calculations[0][i]['image']
        gt = fold_calculations[0][i]['gt']
        predictions = numpy.asarray([fold_calculations[fold][i]['predicted'] for fold in range(len(fold_calculations))])
        prediction = numpy.bincount(predictions).argmax()
        result.append({
            'image': image,
            'gt': gt,
            'predicted': prediction,
        })
        list_prediction.append(prediction)
        list_gt.append(gt)

    result = pandas.DataFrame(result)
    result.to_excel(args.save_path+'/results/prediction_test.xlsx', index=False)



    performance = []
    list_prediction = numpy.asarray(list_prediction)
    list_gt = numpy.asarray(list_gt)
    
    # Accuracy
    correct = numpy.sum((list_prediction==list_gt)*1.0)
    amount = len(list_gt)*1.0
    accuracy = correct/amount
    performance.append({
        'item':'Accuracy',
        'Value': accuracy,
        'Notes': str(round(correct))+ ' / '+str(round(amount))
        })


    

    performance = pandas.DataFrame(performance)
    cwd = os.path.dirname(os.path.realpath(__file__))+'/'
    performance.to_excel(cwd+'/results/Performance_test_set.xlsx', index=False)



            
# python train_bigdata_multiclass.py --im_root '/home/lbwdruid/Desktop/Liver_Ultrasound/0000.data/0006.bigdata/' --yaml_file  'hyper-parameters-bigdata.yml' --datasheet_root '/home/lbwdruid/Desktop/Liver_Ultrasound/0000.data/0006.bigdata/lists/entropy/' --save_path './' --fold 2 --gpu 3
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', type=str, help="path to yaml file for configuration", default='hyper-parameters.yml')
    parser.add_argument('--save_path', type=str, help="path to where you want checkpoints and logs save to", default='hyper-parameters.yml')
    parser.add_argument('--datasheet_root', type=str, help="path to folder with datasheets", default='./')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu ids to use')
    args = parser.parse_args()

    run_test(args)

































