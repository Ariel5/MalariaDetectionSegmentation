import numpy
import json
import scipy
import copy

def read_json(path):
    with open(path, 'r') as f:
        result = json.load(f)
    return result


# config.data.filter.method in ['None', 'Entropy', 'quality']
# config.data.filter.threshold in [0.415, 0.8566]
def clean_labels(list_data, list_quality, config):
    result = []
    field = config.data.predict
    for i in range(len(list_data)):
        try:
            if list_data[i][field]<0:
                continue
            result.append(list_data[i])
        except:
            continue

    if config.data.filter.method=='None':
        return result
    else:
        return __clean_labels_with_options(result, list_criteria=result, config=config)

# 2 options: [a] clean with quality; [b] clean with entropy
def __clean_labels_with_options(list_data, list_criteria, config):
    result = []
    label_field = config.data.filter.method
    threshold = config.data.filter.threshold
    for i in range(len(list_data)):
        if list_criteria[i][label_field]<threshold:
            continue
        result.append(list_data[i])
    return result



def convert_labels_to_multiclass(datalist, config):
    # if config.model.category=='frank_hall':
    return __convert_labels_to_multiclass_frankhall(datalist, config)
    # elif config.model.category=='vanilla':
        # return __convert_labels_to_multiclass_vanilla(datalist, config)
    # return None
  

def __convert_labels_to_multiclass_frankhall(datalist, config):
    new_data_list = []
    field = config.data.predict
    for data_dict in datalist:
        new_label = [0.0 for _ in range(len(config.data.thresholds))]
        old_label = 0
        for i in range(len(config.data.thresholds)):
            if data_dict[config.data.predict]>config.data.thresholds[i]:
                new_label[i] = 1.0
                old_label += 1

        data_dict['old_labels'] = old_label
        data_dict[config.data.predict] = numpy.array(new_label)
        new_data_list.append(data_dict)

    return new_data_list

def __convert_labels_to_multiclass_vanilla(datalist, config):
    new_data_list = []
    field = config.data.predict
    for data_dict in datalist:
        new_label = [0.0 for _ in range(len(config.data.thresholds))]
        old_label = 0
        for i in range(len(config.data.thresholds)):
            if data_dict[config.data.predict]>config.data.thresholds[i]:
                new_label[i] = 1.0
                old_label += 1
        data_dict[config.data.predict] = old_label
        new_data_list.append(data_dict)

    return new_data_list


def create_soft_labels(data_list, config):
    num_outputs = len(config.data.thresholds)+1
    predict = config.data.predict
    
    soft_labels = numpy.zeros((num_outputs, num_outputs))
    for i in range(num_outputs):
        dist = []
        for j in range(num_outputs):
            dist.append(-1*abs(i -j))

        soft_labels[i,:] = scipy.special.softmax(dist)

    for cur_dict in data_list:
        cur_label = int(cur_dict['old_labels'])
        cur_label = soft_labels[cur_label,:]
        cur_dict[predict] = cur_label

    return data_list


def filter_field_with_value(data, field, value):
    new_data = [data_dict for data_dict in data if data_dict[field]==value]
    return new_data


def filter_field_with_values(data, field, values):
    new_data = [data_dict for data_dict in data if data_dict[field] in values]
    return new_data













































