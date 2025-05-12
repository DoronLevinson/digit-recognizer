import torch.nn as nn
import torch.nn.functional as F

class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(12)
        
        self.conv2 = nn.Conv2d(12, 24, kernel_size=6, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(24)
        
        self.conv3 = nn.Conv2d(24, 32, kernel_size=6, stride=2, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32 * 7 * 7, 200, bias=False)
        self.bn4 = nn.BatchNorm1d(200)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(200, 11)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)