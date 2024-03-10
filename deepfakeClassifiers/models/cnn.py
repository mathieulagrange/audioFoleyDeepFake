import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pool_size, n_channels):
        self.pool_size = pool_size
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels= input_size, out_channels= hidden_size,stride=1, kernel_size=3, padding=1,dilation=1)
        self.size_pool1 = (n_channels - (pool_size -1) - 1) +1
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels= hidden_size, out_channels= output_size,stride=1, kernel_size=3, padding=1,dilation=1)
        self.drop = nn.Dropout(0.5)
        self.relu2 = nn.ReLU()
        self.size_pool2 = (self.size_pool1 - (pool_size -1) - 1) +1
        self.linear = nn.Linear(in_features= self.size_pool2*output_size,out_features=1)
        

    def forward(self, x):  
        x = self.relu1(self.conv1(x)) # batch_size x hidden_size x 128

        x = nn.MaxPool1d(kernel_size=self.pool_size,stride=1)(x) # batch_size x hidden_size x 126

        x = self.relu2(self.conv2(x)) # batch_size x output_size x 126

        x = nn.MaxPool1d(kernel_size=self.pool_size,stride=1)(x) # batch_size x output_size x 124
        x = self.drop(x)

        x = nn.Flatten()(x)
        x = self.linear(x)  # batch_size x 1


        return x # batch_size x 2