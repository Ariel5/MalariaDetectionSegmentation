#!/bin/sh
DEVICE=0
SAVE_PATH='/mnt/sdg/bowen/ml_final_project/02.step2-classification/binary_classification_with_weights/'
DATASHEET_ROOT='/mnt/sdg/bowen/ml_final_project/data/single_rbc_resized/'
YAML_FILE='/mnt/sdg/bowen/ml_final_project/02.step2-classification/binary_classification_with_weights/hyper-parameters.yml'

python test_on_dev_set.py --yaml_file ${YAML_FILE} --datasheet_root ${DATASHEET_ROOT} --save_path ${SAVE_PATH} --gpu ${DEVICE}