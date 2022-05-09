import json, pandas, os, io

cwd = os.path.dirname(os.path.realpath(__file__))+'/'

suffix = ['Train', 'Dev', 'Test']
folder_images = '/mnt/sdg/bowen/ml_final_project/data/single_rbc_resized/images/'
folder_json = '/mnt/sdg/bowen/ml_final_project/data/single_rbc_resized/'

for topic in suffix:
    result = []
    df = pandas.read_excel(cwd+'DataSplit_'+topic+'.xlsx')
    for i in range(df.shape[0]):
        row = df.iloc[i]
        result.append({
            'im': folder_images+row['image'],
            'label': int(row['label']),
        })

    with open(folder_json+topic+'_image_list.json', 'w') as f:
        json.dump(result, f)















