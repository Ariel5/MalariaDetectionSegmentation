import pickle, os, numpy, tqdm, pandas

cwd = os.path.dirname(os.path.realpath(__file__))+'/'

# Step 1. Read in data (processed from homework1.ipynb)
with open(cwd+'original_data_all_data.pickle', 'rb') as file:
    all_data = pickle.load(file)

with open(cwd+'original_data_x_train_features.pickle', 'rb') as file:
    x_train = pickle.load(file)

# Step 2. Find the index of those data in the training set
data_split_train = []
for i in tqdm.tqdm(range(len(x_train))):
    current = x_train[i]
    for j in range(len(all_data)):
        if numpy.array_equal(current, all_data[j]['X']):
            data_split_train.append({
                'image': all_data[j]['image'],
                'label': all_data[j]['label'],
            })
            all_data.pop(j)
            break

# Step 3. Further split the training set into train+dev
# train:dev:test = 538:134:224 (60:15:25)
data_split_dev = []
amount_dev = round(len(data_split_train)/75.0*15.0)
for i in range(amount_dev):
    data_split_dev.append(data_split_train[0])
    data_split_train.pop(0)

data_split_test = []
for i in range(len(all_data)):
    data_split_test.append({
        'image': all_data[i]['image'],
        'label': all_data[i]['label'],
    })

# Step 4. Change labels from boolean to int (good for image classification)
for i in range(len(data_split_train)):
    if data_split_train[i]['label']:
        data_split_train[i]['label'] = 1
    else:
        data_split_train[i]['label'] = 0

for i in range(len(data_split_dev)):
    if data_split_dev[i]['label']:
        data_split_dev[i]['label'] = 1
    else:
        data_split_dev[i]['label'] = 0

for i in range(len(data_split_test)):
    if data_split_test[i]['label']:
        data_split_test[i]['label'] = 1
    else:
        data_split_test[i]['label'] = 0

# Step 5. Save
data_split_train = pandas.DataFrame(data_split_train)
data_split_dev = pandas.DataFrame(data_split_dev)
data_split_test = pandas.DataFrame(data_split_test)

data_split_train.to_excel(cwd+'DataSplit_Train.xlsx', index=False)
data_split_dev.to_excel(cwd+'DataSplit_Dev.xlsx', index=False)
data_split_test.to_excel(cwd+'DataSplit_Test.xlsx', index=False)












