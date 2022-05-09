from functions_supporting import get_alias_and_folders, read_yaml
import pandas
import math
from copy import deepcopy
from sklearn.model_selection import KFold, train_test_split
import json
import os
from glob import glob
import re


def create_splits(root_dir, batch_list=['batch1', 'batch2', 'batch3', 'batch4']):

    view_group_dict = {'1': 0,
                       '2': 0,
                       '3': 1,
                       '4': 1,
                       '5': 2,
                       '6': 2,
                       '7': 2,
                       '8': 2,
                       '9': 3,
                       '10': 3,
                       '11': 4,
                       '12': 4,
                       '13': 5,
                       '14': 5,
                       'x': -1}

    alias = get_alias_and_folders(root_dir)
    for group in batch_list:
        if group not in alias:
            continue
        path = alias[group]
        config = read_yaml(path+'config.yml')
        key_list = [key for key in config.info_file.keys(
        ) if key != 'name' and config.info_file[key] != -1]

        df = pandas.read_excel(path+config.info_file.name)
        ref_patient_list = []
        for num, patientRecord in df.iterrows():

            new_patient = {
                key: patientRecord[config.info_file[key]] for key in key_list}

            ref_patient_list.append(new_patient)

        kfold = KFold(
            n_splits=5,
            shuffle=True,
            random_state=1)

        for fold, (train, test) in enumerate(kfold.split(ref_patient_list)):
            # the records overlap between folds, so we need to start fresh
            patient_list = deepcopy(ref_patient_list)
            data_training = [patient_list[num] for num in train]

            # we do 1.25, which is an eighth of 80%, which is a tenth of 100% when considering the
            # entire dataset
            [data_training, data_validation] = train_test_split(data_training, test_size=1/8,
                                                                random_state=1, shuffle=True)
            data_test = [patient_list[num] for num in test]

            process_list = {'train': data_training,
                            'test': data_test,
                            'validation': data_validation}
            scanners = ['TOSHIBA', 'SIEMENS', 'PHILIPS']
            for list_name, cur_patients in process_list.items():
                # to store a list indexed by image
                image_full_list = []
                study_full_list = []
                for cur_patient in cur_patients:

                    cur_path = os.path.join(path, cur_patient['col_ID'])
                    # a copy of the original patient_dict so we can reuse for the study_dict
                    orig_cur_patient = deepcopy(cur_patient)
                    # for each scanner
                    for scanner in scanners:
                        scanner_path = os.path.join(cur_path, scanner)
                        if os.path.exists(scanner_path):

                            cur_study = deepcopy(orig_cur_patient)
                            im_files = glob(os.path.join(
                                scanner_path, '*.tif'))
                            im_dict_list = []
                            for im_file in im_files:

                                view = re.findall(r'View-(\d+)', im_file)
                                if not view:
                                    continue
                                view = view[-1]
                                if view not in view_group_dict:
                                    continue
                                view_group = view_group_dict[view]
                                im_dict = {'im': os.path.relpath(im_file, root_dir),
                                           'view': view,
                                           'scanner': scanner,
                                           'view_group': view_group}
                                # create a complete dictionary for the image that holds all
                                # patient data (for the image_list)
                                im_dict_full = deepcopy(im_dict)
                                im_dict_full.update(orig_cur_patient)
                                image_full_list.append(im_dict_full)
                                im_dict_list.append(im_dict)
                            cur_patient[scanner] = {'im_list' : im_dict_list}
                            cur_study['scanner'] = scanner
                            cur_study['im_list'] = im_dict_list
                            study_full_list.append(cur_study)

                with open(os.path.join(path, list_name + '_patient_list_' + str(fold) + '.json'), 'w') as f:
                    json.dump(cur_patients, f)
                with open(os.path.join(path, list_name + '_image_list_' + str(fold) + '.json'), 'w') as f:
                    json.dump(image_full_list, f)
                with open(os.path.join(path, list_name + '_study_list_' + str(fold) + '.json'), 'w') as f:
                    json.dump(study_full_list, f)

def strip_im_list(root):

    for prefix in ['train', 'test', 'validation']:
        for fold in range(5):
            #do both image and patient lists
            for json_type in ['_image_list_']:
                with open(os.path.join(root, prefix + json_type + str(fold) + '.json'), 'r') as f:
                    dict_list = json.load(f)
                for cur_dict in dict_list:
                    if 'im_list' in cur_dict:
                        cur_dict.pop('im_list')
                with open(os.path.join(root, prefix + json_type + str(fold) + '.json'), 'w') as f:
                    json.dump(dict_list, f)


def add_number_to_json(excel_path, column, id_column, new_column, root):
    # excel_path: path to excel file where new field is coming from
    # column: the column name of the excel file of the new field
    # id_column: the column name of the patient id
    # new_column: the name of the new field in the json file
    # the root location of the json files
    df = pandas.read_excel(excel_path)
    new_dict = {}
    for idx, row in df.iterrows():
        new_val = row[column]
        if math.isnan(new_val):
            new_val = -1

        new_dict[row[id_column]] = new_val

    for prefix in ['train', 'test', 'validation']:
        for fold in range(5):
            #do both image and patient lists
            for json_type in ['_patient_list_', '_image_list_', '_study_list_']:
                with open(os.path.join(root, prefix + json_type + str(fold) + '.json'), 'r') as f:
                    dict_list = json.load(f)
                for cur_dict in dict_list:
                    new_val = new_dict[cur_dict['col_ID']]
                    cur_dict[new_column] = new_dict[cur_dict['col_ID']]
                with open(os.path.join(root, prefix + json_type + str(fold) + '.json'), 'w') as f:
                    json.dump(dict_list, f)



def add_roi_to_json(root):
    json_type='_study_list_'
    for prefix in ['train', 'test', 'validation']:
        for fold in range(5):
            with open(os.path.join(root, prefix + json_type + str(fold) + '.json'), 'r') as f:
                dict_list = json.load(f)
            for cur_dict in dict_list:
                for image in cur_dict['im_list']:
                    path = image['im']
                    path_roi = path.replace('.tif', '-roi.bmp')
                    image['roi'] = path_roi
            with open(os.path.join(root, prefix + json_type + str(fold) + '.json'), 'w') as f:
                json.dump(dict_list, f)


   



# add_roi_to_json(root='/home/lbwdruid/Desktop/Liver_Ultrasound/0000.data/0004.fibrosis_fibroscan')






