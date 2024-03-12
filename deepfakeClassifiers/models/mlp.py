import torch
import torch.nn as nn

class MLP_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_Classifier, self).__init__()

        # Batch Normalization layer
        #self.batch_norm = nn.BatchNorm1d(input_size)

        # Two linear layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)  

        self.linear2 = nn.Linear(hidden_size, hidden_size*2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)  


        self.linear3 = nn.Linear(hidden_size*2, hidden_size)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.5)  


        self.linear4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Apply Batch Normalization
        # x = self.batch_norm(x)

        # Forward pass through the layers
        x = self.dropout1(self.relu1(self.linear1(x)))
        x = self.dropout2(self.relu2(self.linear2(x)))
        x = self.dropout3(self.relu3(self.linear3(x)))
        x = self.linear4(x)
        x = nn.Flatten()(x)

        return x

