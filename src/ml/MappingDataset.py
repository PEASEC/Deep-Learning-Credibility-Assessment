from torch.utils.data import Dataset
from typing import Union, Mapping, Sequence, Iterable, List
import torch


class MappingDataset(Dataset):
    """Dataset wrapping mapping types.

    Each sample will be retrieved by indexing sequences.
    This is a more generic version of the TensorDataset

    Arguments:
        *features (MappingType): feature lists that have the same size of the first dimension.
    """

    def __init__(self, *features: Union[Mapping, Sequence]):
        assert all(len(features[0]) == len(feature) for feature in features), \
            "All feature elements must have the same length"
        self.features = features

    def __len__(self):
        return len(self.features[0])

    def __getitem__(self, index):
        return tuple(feature[index] for feature in self.features)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __to(self, device: torch.device, features: Iterable, result: List):
        for feature in features:
            if isinstance(feature, torch.Tensor):
                result.append(feature.to(device))
            elif callable(getattr(feature, "to", None)):
                result.append(feature.to(device))
            elif isinstance(feature, Iterable):
                result.append(self.__to(device, feature, []))
            else:
                result.append(feature)
        return result

    def to(self, device: torch.device):
        return MappingDataset(*self.__to(device, self.features, []))
