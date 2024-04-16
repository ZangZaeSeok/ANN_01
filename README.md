# ANN_01(Neural network first assignment)
## template
- A directory containing code for training and testing the model.
### dataset.py
- Code to extract train.tar and test.tar stored in the dataset, and convert the stored images into PyTorch datasets.
### model.py
- Code for basic LeNet5, CustomMLP, and Improved LeNet5 (with additional LayerNorm2d and Dropout2d) models.
### main.py
- Code to perform performance evaluation for each model in model.py and save the results in the 'result' directory. Additionally, code to conduct experiments with data augmentation by randomly rotating MNIST images from -30 to 30 degrees ten times to enhance the performance of Improved LeNet5.


## dataset
- A directory where files that are not actually included in this repository, but are needed for training and testing the model, should be stored.

## result
### Default LeNet5 Result
- LeNet5_train_loss.npy
- LeNet5_test_loss.npy
- LeNet5_train_acc.npy
- LeNet5_test_acc.npy
### CustomMLP Result
- CustomMLP_train_loss.npy
- CustomMLP_test_loss.npy
- CustomMLP_train_acc.npy
- CustomMLP_test_acc.npy
### Improved LeNet5 Result
- LeNet5_Imp_train_loss.npy
- LeNet5_Imp_test_loss.npy
- LeNet5_Imp_train_acc.npy
- LeNet5_Imp_test_acc.npy
### Improved LeNet5 with Augmentation Result
- LeNet5_Imp_Augment_train_loss.npy
- LeNet5_Imp_Augment_test_loss.npy
- LeNet5_Imp_Augment_train_acc.npy
- LeNet5_Imp_Augment_test_acc.npy

## report
### report.ipynb
- A Jupyter notebook file containing the actual execution results of each code and the results related to the assignment requirements.
