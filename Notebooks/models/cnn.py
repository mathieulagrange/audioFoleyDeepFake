import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels= 1, out_channels= 64,stride=1, kernel_size=3, padding=1,dilation=1)
        self.conv2 = nn.Conv1d(in_channels= 64, out_channels= 64,stride=1, kernel_size=3, padding=1,dilation=1)
        # self.conv3 = nn.Conv1d(in_channels= 64, out_channels= 128,stride=1, kernel_size=3, padding=1,dilation=1)
        # self.conv4 = nn.Conv1d(in_channels= 128, out_channels= 64,stride=1, kernel_size=3, padding=1,dilation=1)
        self.bn1 = nn.BatchNorm1d(64)
        # self.bn2 = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.1)
        self.fc = nn.Linear(in_features= 64*128,out_features=1)

    def forward(self, x):  
        x = F.relu(self.conv1(x)) # 32 x 64 x 128
        x = F.relu(self.bn1(self.conv2(x))) # 32 x 64 x 128
        x = self.drop(x)
        x = torch.flatten(x, 1)  # 32 x 64*128
        x = self.fc(x) # 32 x 1
        return x # 32 x 2