import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List
from torch.nn.utils.rnn import pack_sequence


class TorchList(list):
    def __init__(self, *args):
        super().__init__(*args)

    def to(self, *args):
        pass


class TimelineRNN(nn.Module):
    def __init__(self, in_features=49, out_features=1, embedding_size=50,
                 hidden_size=32, rnn_hidden_size=32, bidirectional=True,
                 dropout_rate=0, rnn_dropout_rate=0):
        super(TimelineRNN, self).__init__()

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

        self.timeline_rnn = nn.RNN(embedding_size, rnn_hidden_size, 1, batch_first=True,
                                   bidirectional=bidirectional, dropout=rnn_dropout_rate)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, normal_features, text_features, timeline_text_features, timeline_elements: TorchList[int]):
        output, h_n = self.rnn(text_features)
        # h_n is the last layer of the rnn and has shape
        # (num_layers * num_directions, batch, hidden_size)
        # However, we want to have (batch, hidden_size * num_layers * num_directions)
        size = h_n.size()
        h_n = h_n.transpose(0, 1).reshape((size[1], size[0] * size[2]))

        output2, h2_n = self.timeline_rnn(timeline_text_features)
        size2 = h2_n.size()
        h2_n = h2_n.transpose(0, 1).reshape((size2[1], size2[0] * size2[2]))
        # Now lets split h2_n according to the length of the timeline_elements
        h2_splitted = torch.split(h2_n, timeline_elements)
        h2_mean = torch.stack([torch.mean(x, dim=0) for x in h2_splitted])
        h2_min = torch.stack([torch.min(x, dim=0)[0] for x in h2_splitted])
        h2_max = torch.stack([torch.max(x, dim=0)[0] for x in h2_splitted])

        # Concat everything
        x = torch.cat((h_n, normal_features, h2_mean, h2_min, h2_max), dim=1)

        # Put it into our neural network
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x

    @staticmethod
    def collate_fn(data: List[Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]]):
        [normal_features, text_features, timeline_text_features, labels] = zip(*data)

        indices = np.flip(np.argsort([len(f) for f in text_features]))

        # Faster than building numpy arrays:
        normal_features = [normal_features[i] for i in indices]
        text_features = [text_features[i] for i in indices]
        labels = [labels[i] for i in indices]
        timeline_text_features = [timeline_text_features[i] for i in indices]

        # First retrieve length of each component in the list
        timeline_elements = torch.tensor([len(e) for e in timeline_text_features])
        # Then flatten it
        timeline_text_features = [item for sublist in timeline_text_features for item in sublist]

        normal_features = torch.stack(normal_features)
        text_features = pack_sequence(text_features)
        timeline_text_features = pack_sequence(timeline_text_features, enforce_sorted=False)
        labels = torch.stack(labels)

        return normal_features, text_features, timeline_text_features, TorchList(timeline_elements), labels

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
