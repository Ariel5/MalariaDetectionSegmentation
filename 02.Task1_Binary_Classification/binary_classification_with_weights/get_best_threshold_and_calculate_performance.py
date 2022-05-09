import os, pandas, numpy

cwd = os.path.dirname(os.path.realpath(__file__))+'/'

result = []
for fold in range(5):
    df = pandas.read_excel(cwd+'/results/prediction_dev_f'+str(fold)+'.xlsx')
    list_prediction = numpy.asarray(df['predicted'].tolist())
    list_gt = numpy.asarray(df['gt'].tolist())


    def calculate_accuracy_with_threshold(prediction, gt, threshold):
        predict = (prediction>threshold)*1.0

        amount = len(list_gt)*1.0
        correct = numpy.sum((predict==gt)*1.0)
        accuracy = correct/amount
        
        return accuracy


    threshold = -10
    best_accuracy = 0
    best_threshold = -10

    while threshold<10:
        current_accuracy = calculate_accuracy_with_threshold(list_prediction, list_gt, threshold)
        if current_accuracy>best_accuracy:
            best_accuracy=current_accuracy
            best_threshold = threshold
        threshold+=0.01
    
    result.append({
        'fold': fold,
        'dev_accuracy': best_accuracy,
        'threshold': best_threshold
    })

result = pandas.DataFrame(result)
result.to_excel(cwd+'/results/thresholds.xlsx', index=False)


















