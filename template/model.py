import torch
import torch.nn as nn

class LeNet5_Imp(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.ConvEncoder = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )

        self.Fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.LeakyReLU(),
            nn.Linear(120, 84),
            nn.LeakyReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, img):
        img = self.ConvEncoder(img)
        output = self.Fc(img)
        return output

class CustomMLP(nn.Module):
    def __init__(self):
        super(CustomMLP, self).__init__()

        self.MLP = nn.Sequential(
            nn.Linear(28*28, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 10),
            nn.LogSoftmax(-1)
        )

    def forward(self, img):
        img = img.reshape(img.shape[0], -1)
        output = self.MLP(img)
        return output

# Improved Version
class LeNet5_Imp(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.ConvEncoder = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            # two regularization techniques
            nn.Dropout2d(p=0.5),
            nn.LayerNorm(normalized_shape=[6, 24, 24]),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5),
            # two regularization techniques
            nn.Dropout2d(p=0.5),
            nn.LayerNorm(normalized_shape=[16, 8, 8]),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )

        self.Fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.LeakyReLU(),
            nn.Linear(120, 84),
            nn.LeakyReLU(),
            nn.Linear(84, 10),
            nn.LogSoftmax(-1)
        )

        self.weight_init(self)

    def weight_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)

    def forward(self, img):
        img = self.ConvEncoder(img)
        output = self.Fc(img)
        return output