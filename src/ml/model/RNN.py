import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List
from torch.nn.utils.rnn import pack_sequence


class RNN(nn.Module):
    def __init__(self, in_features=49, out_features=1, embedding_size=50,
                 hidden_size=32, rnn_hidden_size=32, bidirectional=True,
                 dropout_rate=0, rnn_dropout_rate=0):
        super(RNN, self).__init__()

        rnn_output_size = rnn_hidden_size
        if bidirectional:
            rnn_output_size *= 2

        # We concat the rnn output with the input features before
        # putting it into an fully connected layer
        in_features += rnn_output_size

        self.fc1 = nn.Linear(in_features=in_features,
                             out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size,
                             out_features=out_features)

        self.rnn = nn.RNN(embedding_size, rnn_hidden_size, 1, batch_first=True,
                          bidirectional=bidirectional, dropout=rnn_dropout_rate)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, normal_features, text_features):
        output, h_n = self.rnn(text_features)
        # h_n is the last layer of the rnn and has shape
        # (num_layers * num_directions, batch, hidden_size)
        # However, we want to have (batch, hidden_size * num_layers * num_directions)
        size = h_n.size()
        h_n = h_n.transpose(0, 1).reshape((size[1], size[0] * size[2]))

        # Use last layer of rnn and concat remaining features
        x = torch.cat((h_n, normal_features), dim=1)

        # Put it into our neural network
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x

    @staticmethod
    def collate_fn(data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        features = list(zip(*data))
        assert len(features) > 1, "Cannot parse text features"

        for i in range(len(features)):
            # Text-features
            if i == 1:
                features[i] = pack_sequence(features[i], enforce_sorted=False)
            else:
                features[i] = torch.stack(features[i])

        return tuple(features)

    def load_rnn_state(self, file_path: str):
        remote_state = torch.load(file_path, map_location="cpu")
        if "state_dict" in remote_state:
            remote_state = remote_state["state_dict"]

        local_state = self.state_dict().copy()
        for key in local_state:
            if key.startswith("rnn."):
                local_state[key] = remote_state[key]

        self.load_state_dict(local_state)

# # Test
# input1 = (
#     torch.tensor([5, 9, 25, 8]),
#     torch.tensor([
#         [99, 102, 108, 29, 28],
#         [99, 102, 108, 29, 28],
#         [99, 102, 108, 29, 28]
#     ]),
#     torch.tensor(1),
# )
#
# input2 = (
#     torch.tensor([0, 9, 1, 2]),
#     torch.tensor([
#         [99, 102, 108, 29, 28],
#         [99, 102, 108, 29, 28]
#     ]),
#     torch.tensor(2)
# )
#
# input3 = (
#     torch.tensor([0, 9, 1, 20]),
#     torch.tensor([
#      [99, 102, 108, 29, 28],
#      [99, 102, 108, 29, 28],
#      [99, 102, 108, 29, 28],
#      [99, 102, 108, 29, 28]
#     ]),
#     torch.tensor(3)
# )
#
# print(RNN.collate_fn([input1, input2, input3]))
