import pandas, json

# Step 1. Create a look up table for stages
df = pandas.read_excel('/mnt/sdg/bowen/ml_final_project/01.step1-construct-dataset-according-to-hw1/staging-labels.xlsx')

lookup_table = {}
for i in range(df.shape[0]):
    row = df.iloc[i]
    stage = 0
    if row['Stage']=='R':
        stage = 1
    elif row['Stage']=='T':
        stage = 2
    elif row['Stage']=='S':
        stage = 3
    elif row['Stage']=='G':
        stage = 4
    lookup_table[row['Name']] = stage

# Step 2. Modify existing json files
folder_target = '/mnt/sdg/bowen/ml_final_project/data/single_rbc_resized/staging-labels/'
list_jsons = [
    '/mnt/sdg/bowen/ml_final_project/data/single_rbc_resized/Dev_image_list_0.json',
    '/mnt/sdg/bowen/ml_final_project/data/single_rbc_resized/Dev_image_list_1.json',
    '/mnt/sdg/bowen/ml_final_project/data/single_rbc_resized/Dev_image_list_2.json',
    '/mnt/sdg/bowen/ml_final_project/data/single_rbc_resized/Dev_image_list_3.json',
    '/mnt/sdg/bowen/ml_final_project/data/single_rbc_resized/Dev_image_list_4.json',
    '/mnt/sdg/bowen/ml_final_project/data/single_rbc_resized/Test_image_list.json',
    '/mnt/sdg/bowen/ml_final_project/data/single_rbc_resized/Train_image_list_0.json',
    '/mnt/sdg/bowen/ml_final_project/data/single_rbc_resized/Train_image_list_1.json',
    '/mnt/sdg/bowen/ml_final_project/data/single_rbc_resized/Train_image_list_2.json',
    '/mnt/sdg/bowen/ml_final_project/data/single_rbc_resized/Train_image_list_3.json',
    '/mnt/sdg/bowen/ml_final_project/data/single_rbc_resized/Train_image_list_4.json',
]


for json_file in list_jsons:
    file_name = json_file.split('/')[-1]
    with open(json_file) as f:
        data = json.load(f)
    
    for i in range(len(data)):
        image_name = data[i]['im'].split('/')[-1].split('.')[0]
        data[i]['label'] = lookup_table[image_name]

    with open(folder_target+file_name, 'w') as f:
        json.dump(data, f)





































