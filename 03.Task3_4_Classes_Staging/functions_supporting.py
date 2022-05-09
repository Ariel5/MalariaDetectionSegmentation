import yaml
import json
from easydict import EasyDict as edict
import os
import shutil
import pandas
import sklearn.model_selection
import torchvision
import sklearn

def show_image_array(imagearray):
    PIL.Image.fromarray(imagearray).show()

def show_image_tensor(tensor):
    # show_image_array(tensor.numpy().transpose(1, 2, 0))
    torchvision.transforms.ToPILImage()(tensor).convert("RGB").show()

def read_yaml(path):
    try:
        with open(path, 'r') as f:
            file = edict(yaml.load(f, Loader=yaml.FullLoader))
        return file
    except:
        return None
        
def get_alias_and_folders(data_path):
    alias = {}
    for r, d, f in os.walk(data_path):
        for file in f:
            if '.yml' in file:
                c = read_yaml(r+'/'+file)
                if c:
                    alias[c.alias] = r+'/'
    return alias


def get_data_jsons(config, fold, data_path, prefix='_image_list_'):

    alias = get_alias_and_folders(data_path)

    patient_list = []
    looplist = config.data.train
    train_json = []
    validation_json = []
    test_json = []
    for group in looplist:
        if group not in alias:
            continue
        path = alias[group]
        f_path = os.path.join(path, 'train'+ prefix + str(fold) + '.json')
        with open(f_path, 'r') as f:
            cur_train_json = json.load(f)
            train_json.extend(cur_train_json)
        f_path = os.path.join(path, 'test'+ prefix + str(fold) + '.json')
        with open(f_path, 'r') as f:
            cur_test_json = json.load(f)
            test_json.extend(cur_test_json)
        f_path = os.path.join(path, 'validation'+ prefix + str(fold) + '.json')
        with open(f_path, 'r') as f:
            cur_validation_json = json.load(f)
            validation_json.extend(cur_validation_json)

    return train_json, validation_json, test_json

def prepare_data(config, K, test_ratio=0.1):
    # Step 1. Prepare: find all data with alias
    alias = get_alias_and_folders(config.data_path)

    patient_list = []
    if K == 0:
        looplist = config.test.data
    else:
        looplist = config.data.train
    for group in looplist:
        if group not in alias:
            continue
        path = alias[group]
        c = read_yaml(path+'config.yml')
        df = pandas.read_excel(path+c.info_file.name)
        for i in range(df.shape[0]):
            patientRecord = df.iloc[i, :]
            patientId = patientRecord[c.info_file.col_ID-1]
            patientPath = path+str(patientId)+'/'

            truth = []
            for head in config.data.predict:
                try:
                    truth.append(patientRecord[c.info_file[head]-1])
                except:
                    truth.append(None)

            newPatient = Patient(ID=patientId, path=patientPath, truth=truth,
                                 truth_columns=config.data.predict, config=config)
            patient_list.append(newPatient)


    data_training = []
    data_validation = []
    data_testing = []

    patient_list = sklearn.utils.shuffle(patient_list, random_state=1)

    for _ in range(int(len(patient_list)*test_ratio)):
        data_testing.append(patient_list.pop(-1))

    # For testing dataset, K is 0
    if K == 0:
        return patient_list, None
    # Else construct K fold data for training and validation
    kfold = sklearn.model_selection.KFold(
        n_splits=config.data.validation_fold,
        shuffle=config.data.shuffle,
        random_state=1)

    for train, test in kfold.split(patient_list):
        data_training.append([patient_list[num] for num in train])
        data_validation.append([patient_list[num] for num in test])

    return data_training, data_validation, data_testing
