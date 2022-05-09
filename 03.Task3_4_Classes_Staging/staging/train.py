import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

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
from dlt.common.controllers import ModelSaver, RollingAverageModelSaver
from dlt.common.layers.batch_wrappers import MetricWrapper


def run_train(args):
    ################################################################################################################
    ##### Environment Setup
    ################################################################################################################
    torch.cuda.set_device(args.gpu)
    config = read_yaml(args.yaml_file)
    
    # Create context
    context = Context()
    context['config'] = config

    ################################################################################################################
    ##### Data Preparation
    ################################################################################################################
    data_training = functions_data_preprocessing.read_json(args.datasheet_root + 'Train_image_list_'+str(args.fold)+'.json')
    data_validation = functions_data_preprocessing.read_json(args.datasheet_root + 'Dev_image_list_'+str(args.fold)+'.json')

    # data_training = functions_data_preprocessing.convert_labels_to_multiclass(data_training, config)
    # data_validation = functions_data_preprocessing.convert_labels_to_multiclass(data_validation, config)
    
    # Step 1.1 Define transforms
    transforms_training = transforms.Transforms_Study(
        im_root = config.data.path,
        new_size = config.data.patch_size,
        im_fields = ['im'],
        label_field = [config.data.predict],
        image_augmention = [
            torchvision.transforms.ColorJitter(brightness=.1, contrast=.1),
            torchvision.transforms.RandomAffine(degrees=10, scale=(.9,1.1), shear=2, resample=PIL.Image.BILINEAR),
            ],
        )
    transforms_validation = transforms.Transforms_Study(
        im_root = config.data.path,
        new_size = config.data.patch_size,
        im_fields = ['im'],
        label_field = [config.data.predict],
        image_augmention = None,
        )

    # Step 1.2 Construct datasets, dataloaders
    dataset_training = BaseDataset(
        listdict = data_training,
        transforms = transforms_training,
        )
    dataset_validation = BaseDataset(
        listdict = data_validation, 
        transforms = transforms_validation,
        )


    dataloader_training = torch.utils.data.DataLoader(
        dataset_training,
        batch_size = config.solver.batch_size,
        shuffle = False,
        num_workers = config.solver.num_workers,
        pin_memory = True,
        )   
    dataloader_validation = torch.utils.data.DataLoader(
        dataset_validation,
        batch_size = config.solver.batch_size, 
        shuffle = False,
        num_workers = config.solver.num_workers,
        pin_memory = True,
        )

    '''
    # For debugging only:
    it = iter(dataloader_training)
    first = next(it)
    print (first)
    
    print (len(dataloader_training))
    print (len(dataloader_validation))
    '''    
    
    # Step 1.3 Set up context data
    context['component.train_dataset'] = dataset_training
    context['component.val_dataset'] = dataset_validation
    context['component.train_loader'] = dataloader_training
    context['component.val_loader'] = dataloader_validation

    # Step 1.4 Move data to cuda
    cuda_mover = PreIterTransform(
        dlt.common.transforms.MoveToCuda(
            fields=['var.batch_data.im', 'var.batch_data.'+config.data.predict],
            )
        )
    context['component.cuda_mover'] = cuda_mover

    ###############################################################################################
    ###### Step 2. Model Preparation
    ###############################################################################################

    # Step 2.0 Set up model
    model = network.WrappedNetwork(
         config = config,
         im_key = 'im',
         ).cuda()
    context['component.model'] = model

    # Step 2.1 Set up loss function
    if 'criterion_weights' not in config.solver:
        config.solver.criterion_weights = None
    loss = MetricWrapper(
        label_field = config.data.predict,
        prediction_field = 'output',
        layer_instance = criteria[config.solver.criterion](config.solver.criterion_weights),
        )
    context['component.loss'] = loss

    # Step 2.2 Set up loss optimizer
    optimizer = torch.optim.SGD(
        context['component.model'].parameters(),
        lr = config.solver.learning_rate,
        weight_decay = config.solver.weight_decay,
        )
    context['component.optimizer'] = optimizer

    ################################################################################################################
    ##### Tensorboard Setup
    ################################################################################################################
    context['component.metric.avg_loss'] = OnlineAverageLoss()

    # create a tensorboard logger
    workspace = os.path.join(args.save_path, 'runs', config.study.name)
    init_workspace(workspace)
    tensorboard_monitor = TensorboardLogger(workspace)
    context['component.monitor.tensorboard'] = tensorboard_monitor

    # # add dependencies
    tensorboard_monitor.add_dependency('callback_post_val_epoch', 'component.metric.avg_loss')
    tensorboard_monitor.add_dependency('callback_post_train_epoch', 'component.metric.avg_loss')
    tensorboard_monitor.add_to_post_val_epoch('var.avg_val_loss', 'Val Loss - F'+str(args.fold))
    tensorboard_monitor.add_to_post_train_epoch('var.avg_train_loss', 'Train Loss- F'+str(args.fold))


    ################################################################################################################
    ##### Console Monitors Setup
    ################################################################################################################

    # set up a per-iter console monitor
    console_monitor = TQDMConsoleLogger(
        extra_train_logs=('var.avg_train_loss', 'Avg. Train Loss %.3f'),
        extra_val_logs=('var.avg_val_loss', 'Avg. Val Loss %.3f'),
        )

    # Console monitor depends on the average loss, add the dependencies. 
    console_monitor.add_dependency('callback_post_train_iter', 'component.metric.avg_loss')
    console_monitor.add_dependency('callback_post_val_iter', 'component.metric.avg_loss')
    context['component.monitor.console'] = console_monitor

    # set up epoch  mintor for JH metric
    print_monitor_tags = [('var.avg_train_loss', 'Train Loss %.3f')]
    epoch_monitor = PrintEpochLogger(val_logs=print_monitor_tags, train_logs=print_monitor_tags)

    epoch_monitor.add_dependency('callback_post_val_epoch', 'component.metric.avg_loss')
    epoch_monitor.add_dependency('callback_post_train_epoch', 'component.metric.avg_loss')
    context['component.monitor.epoch_console'] = epoch_monitor
    
    ################################################################################################################
    ##### Model Saver Setup
    ################################################################################################################
    ckpt_path = os.path.join(args.save_path, 'ckpts', config.study.name+'-F'+str(args.fold))
    os.makedirs(ckpt_path, exist_ok=True)

    # it tracks the best average JH, expecting higher values to be better
    model_saver = RollingAverageModelSaver(
        output_dir = ckpt_path,
        metric_field = 'var.avg_val_loss',
        lower_is_better = True,
        rolling_avg_num = 5,
        )
    
    # make sure we have the average AUC calculated before we do model saving
    model_saver.add_dependency('callback_post_val_epoch', 'component.metric.avg_loss')
    context['component.controller'] = model_saver


    ###############################################################################################
    ###### Create trainer and start training ######################################################
    ###############################################################################################
    trainer = SupervisedTrainer(context, 1, config.solver.get('epochs'))

    trainer.run()




# python train_bigdata_multiclass.py --im_root '/home/lbwdruid/Desktop/Liver_Ultrasound/0000.data/0006.bigdata-new/' --yaml_file  'hyper-parameters-bigdata.yml' --datasheet_root '/home/lbwdruid/Desktop/Liver_Ultrasound/0000.data/0006.bigdata-new/lists_new/' --save_path './' --fold 2 --gpu 3
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', type=str, help="path to yaml file for configuration", default='hyper-parameters.yml')
    parser.add_argument('--save_path', type=str, help="path to where you want checkpoints and logs save to", default='hyper-parameters.yml')
    parser.add_argument('--datasheet_root', type=str, help="path to folder with datasheets", default='./')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu ids to use')
    parser.add_argument('--fold', type=int, default=-1, help='fold')
    args = parser.parse_args()

    run_train(args)

































