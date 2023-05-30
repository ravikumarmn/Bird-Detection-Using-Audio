import torch
import torch.nn as nn


class BirdClassifier(nn.Module):
    def __init__(self, num_classes=5,l2_lambda=0.01):
        super(BirdClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16 * 56 * 56, 128)
        self.relu3 = nn.ReLU()

        self.dropout = nn.Dropout(p=0.2)  # Adding dropout layer

        self.fc2 = nn.Linear(128, num_classes)
        self.l2_lambda = l2_lambda

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = self.fc1(x)
        x = self.relu3(x)

        x = self.dropout(x)  # Applying dropout

        x = self.fc2(x)

        return x

    def l2_regularization_loss(self):
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param, 2)
        return self.l2_lambda * l2_loss
