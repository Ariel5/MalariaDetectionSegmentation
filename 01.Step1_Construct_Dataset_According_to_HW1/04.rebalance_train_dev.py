import pandas, random, json

df0 = pandas.read_excel('/mnt/sdg/bowen/ml_final_project/01.step1-construct-dataset-according-to-hw1/DataSplit_Train.xlsx')
df1 = pandas.read_excel('/mnt/sdg/bowen/ml_final_project/01.step1-construct-dataset-according-to-hw1/DataSplit_Dev.xlsx')
folder_images = '/mnt/sdg/bowen/ml_final_project/data/single_rbc_resized/images/'
folder_json = '/mnt/sdg/bowen/ml_final_project/data/single_rbc_resized/'

list_cases = []
for i in range(df0.shape[0]):
    row = df0.iloc[i]
    list_cases.append({
        'im': folder_images+row['image'],
        'label': int(row['label']),
    })
for i in range(df1.shape[0]):
    row = df0.iloc[i]
    list_cases.append({
        'im': folder_images+row['image'],
        'label': int(row['label']),
    })

random.seed(10000)
random.shuffle(list_cases)
parts = []
for i in range(5):
    index_0 = len(list_cases)//5*i
    if i<4:
        index_1 = index_0+len(list_cases)//5
    else:
        index_1 = len(list_cases)
    parts.append(list_cases[index_0:index_1])


for i in range(5):
    index_train = [item for item in [0,1,2,3,4] if item!=i]
    list_dev = parts[i]
    list_train = []
    for j in index_train:
        list_train.extend(parts[j])
    
    with open(folder_json+'Train_image_list_'+str(i)+'.json', 'w') as f:
        json.dump(list_train, f)
    with open(folder_json+'Dev_image_list_'+str(i)+'.json', 'w') as f:
        json.dump(list_dev, f)










