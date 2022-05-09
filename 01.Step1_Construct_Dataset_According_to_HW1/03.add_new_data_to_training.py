import os, pandas, shutil

folder_old = '/mnt/sdg/bowen/ml_final_project/data/single_rbc_resized/images/'
folder_new = '/mnt/sdg/bowen/ml_final_project/data/single_rbc_resized/new-cohort/'

# Step 1. only look at new images
list_images_old = os.listdir(folder_old)
list_images_new = os.listdir(folder_new)
list_images_new = [item for item in list_images_new if item not in list_images_old]

# Step 2. Load old DataSplit_Train.xlsx
df = pandas.read_excel('/mnt/sdg/bowen/ml_final_project/01.step1-construct-dataset-according-to-hw1/DataSplit_Train.xlsx')
result = []
for i in range(df.shape[0]):
    row = df.iloc[i]
    result.append({
        'image':row['image'],
        'label':row['label']
    })

# Step 3. Read in annotations for new data
df = pandas.read_excel('/mnt/sdg/bowen/ml_final_project/01.step1-construct-dataset-according-to-hw1/parasite_labeling_4.28.22.xlsx')
label_dict = {}
for i in range(df.shape[0]):
    row = df.iloc[i]
    label_dict[row['Name']+'.jpg'] = row['infected']

# Step 4. Add new data to DataSplit_Train.xlsx
for item in list_images_new:
    if label_dict[item]=='Y':
        label = 1
    else:
        label = 0
    result.append({
        'image':item,
        'label':label
    })

# Step 5. Save df
result = pandas.DataFrame(result)
result.to_excel('/mnt/sdg/bowen/ml_final_project/01.step1-construct-dataset-according-to-hw1/DataSplit_Train.xlsx')

# Step 6. Save images
for item in list_images_new:
    path_0 = folder_new+item
    path_1 = folder_old+item
    shutil.copyfile(path_0, path_1)


# Step 7. run 02.create_jsons.py















