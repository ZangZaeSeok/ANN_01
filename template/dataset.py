import os
import numpy as np
import random
import torch
import tarfile
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class MNIST(Dataset):
    def __init__(self, data_dir):
        super(MNIST, self).__init__()
        self.fileNames = []                         # Image file Names List
        self.targets = []                           # Image Label List
        self.data_dir = data_dir                    # Root dir

        # all values should be in a range of [0,1]
        # Substract mean of 0.1307, and divide by std 0.3081
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        for filename in os.listdir(self.data_dir):
            if 'png' in filename:
                self.fileNames.append(os.path.join(self.data_dir, filename))
                label = int(filename.split("_")[1].split(".")[0])
                self.targets.append(label)

    def __len__(self):
        return len(self.fileNames)

    def __getitem__(self, idx):
        img = Image.open(self.fileNames[idx]).convert('L')
        img = self.transform(img)
        label = self.targets[idx]

        return img, label

if __name__ == '__main__':
    data_dir = '../data/test.tar'
    root, _ = os.path.splitext(data_dir)
    tar = tarfile.open(data_dir, 'r')
    data_dir = os.path.dirname(data_dir)
    tar.extractall(data_dir)
    
    data_dir = '../data/train.tar'
    root, _ = os.path.splitext(data_dir)
    tar = tarfile.open(data_dir, 'r')
    data_dir = os.path.dirname(data_dir)
    tar.extractall(data_dir)
    
    trainset = MNIST('../data/train')
    testset = MNIST('../data/test')
    
    print('MNIST Train Dataset Description')
    print(f'Sample Dataset Len: {len(trainset)}')
    print(f'Sample image shape: {trainset[0][0].shape}, Sample label: {trainset[0][1]}')
    print(f'Sample image mean value: {torch.mean(trainset[0][0].reshape(-1))}, Sample image std value{torch.std(trainset[0][0].reshape(-1))}')
    
    print('Sample train image example')
    plt.imshow(trainset[0][0].reshape(28, 28))
    plt.show()
    
    print()
    
    print('MNIST Test Dataset Description')
    print(f'Sample Test Dataset Len: {len(testset)}')
    print(f'Sample image shape: {testset[0][0].shape}, Sample label: {testset[0][1]}')
    print(f'Sample image mean value: {torch.mean(testset[0][0].reshape(-1))}, Sample image std value{torch.std(testset[0][0].reshape(-1))}')
    
    print('Sample test image example')
    plt.imshow(testset[0][0].reshape(28, 28))
    plt.show()


