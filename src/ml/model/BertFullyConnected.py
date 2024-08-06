import torch
import torch.nn as nn
import random

random.seed(42)


class NeuralNet(nn.Module):
    def __init__(self, in_features=49, out_features=1, dropout=0.1, bert_drop_p=0.8, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features,
                             out_features=hidden_size if hidden_size > 0 else out_features)

        self.sigmoid = nn.Sigmoid()

        if hidden_size > 0:
            self.tanh = nn.Tanh()
            self.fc2 = nn.Linear(in_features=hidden_size,
                                 out_features=out_features)
        else:
            self.tanh = None
            self.fc2 = None

        self.dropout = nn.Dropout(p=dropout)
        self.bert_drop_p = bert_drop_p

    def forward(self, feature_vector, bert_vector):
        if self.training and random.random() <= self.bert_drop_p:
            bert_vector = torch.zeros(bert_vector.size()).to(bert_vector.device)

        out = torch.cat((bert_vector, feature_vector), dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        if self.fc2 is not None:
            out = self.tanh(out)
            out = self.dropout(out)
            out = self.fc2(out)

        out = self.sigmoid(out)
        return out
