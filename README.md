# MalariaDetectionSegmentation
CS.620.675 Machine Learning final project with Sean Yanik and Bowen Li

## Data Preparation
Python scripts in folder '01.Step1_Construct_Dataset_According_to_HW1' should were run sequentially to construct the 5-fold cross-validation json dataset for Task 1 and Task 3.

## Task 1: Binary Classification
The goal for task 1 is to classify a single red blood cell as 'infected' (1) or 'uninfected' (0).

We used DenseNet121 as the CNN backbone to do the binary classification, which could be found in '02.Task1_Binary_Classification/binary_classification_with_weights/network.py'.

Because of the limitation of data volume, we had to run a 5-fold cross-validation, and assemble the output of each folds as the final prediction for the test set.

To train the algorithm, run '02.Task1_Binary_Classification/binary_classification_with_weights/01.start_training.sh'. An example to run this bash script is 'bash 01.start_training.sh 0', where 0 indicates the first fold of data.

To decide the best operating point based on the development set, two scripts should be ran sequentially: '02.Task1_Binary_Classification/binary_classification_with_weights/02.test_dev.sh', and '02.Task1_Binary_Classification/binary_classification_with_weights/03.get_thresholds.sh'. Calculated thresholds could be found in  '02.Task1_Binary_Classification/binary_classification_with_weights/results/thresholds.xlsx'.

Finally, to test the performance on test set, run '02.Task1_Binary_Classification/binary_classification_with_weights/04.test_test.sh'. Case-wise predictions could be found in '02.Task1_Binary_Classification/binary_classification_with_weights/results/prediction_test.xlsx', and the performance (including accuracy, sensitivity, and specificity) could be found in '02.Task1_Binary_Classification/binary_classification_with_weights/results/Performance_test_set.xlsx'.

## Task 3: 4-Class Staging
The goal for task 3 is to stage a single red blood cell as 'uninfected' (0), or 'R= ring' (1), or 'T= trophozoite' (2), or 'S= Schizont' (3), or 'G= gametocyte' (4).

We used DenseNet121 as the CNN backbone to do the staging classification, which could be found in '03.Task3_4_Classes_Staging/staging/network.py'.

Because of the limitation of data volume, we had to run a 5-fold cross-validation, and assemble the output of each folds as the final prediction for the test set.

To train the algorithm, run '03.Task3_4_Classes_Staging/staging/01.start_training.sh'. An example to run this bash script is 'bash 01.start_training.sh 0', where 0 indicates the first fold of data.

Unlike Task 1, for this task, we optimized the workflow, so threshold need to be calculated based on the development set.

Finally, to test the performance on test set, run '03.Task3_4_Classes_Staging/staging/04.test_test.sh'. Case-wise predictions could be found in '03.Task3_4_Classes_Staging/staging/results/prediction_test.xlsx', and the performance (accuracy) could be found in '03.Task3_4_Classes_Staging/staging/results/Performance_test_set.xlsx'.













