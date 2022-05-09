#!/bin/sh
DEVICE=0
SAVE_PATH='/mnt/sdg/bowen/ml_final_project/03.step3-staging-4classes/staging/'
DATASHEET_ROOT='/mnt/sdg/bowen/ml_final_project/data/single_rbc_resized/staging-labels/'
YAML_FILE='/mnt/sdg/bowen/ml_final_project/03.step3-staging-4classes/staging/hyper-parameters.yml'

python test_on_test_set.py --yaml_file ${YAML_FILE} --datasheet_root ${DATASHEET_ROOT} --save_path ${SAVE_PATH} --gpu ${DEVICE}