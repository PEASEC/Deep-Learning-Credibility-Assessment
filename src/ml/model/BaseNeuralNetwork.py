import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, in_features=49, out_features=1, dropout=0.4, hidden_size=32):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features,
                             out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size,
                             out_features=out_features)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
