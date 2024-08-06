import torch
import time
from torch.utils.data import Dataset
from .utils import load_all_from_dataset
from typing import Tuple


class NeuralNetWrapper:
    def __init__(self, model: torch.nn.Module, device: torch.device = None, collate_fn=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model
        self.device = device
        self.model.to(device)
        self.collate_fn = collate_fn

        self.mean = 0
        self.std = 1

    def set_collate_fn(self, collate_fn):
        self.collate_fn = collate_fn

    def load_all_from_dataset(self, data):
        return load_all_from_dataset(data, self.collate_fn)

    def predict(self, features: Tuple[torch.Tensor, ...], normalize: bool = True) -> torch.Tensor:
        model = self.model.eval()

        if normalize:
            features = self.normalize_tuple(features)

        transformed_features: list = list(features)
        for i in range(len(features)):
            transformed_features[i] = transformed_features[i].to(self.device)

        with torch.no_grad():
            predictions = model(*transformed_features)
            return torch.reshape(predictions, (-1,))

    def predict_dataset(self, dataset: Dataset, normalize: bool = True):
        return self.predict(self.load_all_from_dataset(dataset), normalize)

    def save(self, file_path: str = None, data=None):
        if data is None:
            data = {}

        if file_path is None:
            n = str(round(time.time() * 1000))
            file_path = "build/{}.pt".format(n)

        data["state_dict"] = self.model.state_dict()
        data["mean"] = self.mean
        data["std"] = self.std

        torch.save(data, file_path)

    def load(self, file_path: str = None):
        data = torch.load(file_path, map_location="cpu")
        self.model.load_state_dict(data["state_dict"])
        self.mean = data["mean"]
        self.std = data["std"]

        return data

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor -= self.mean
        tensor /= self.std

        return tensor

    def normalize_tuple(self, tuple_tensor: Tuple[torch.Tensor, ...]):
        primitive_data, *rest = tuple_tensor
        primitive_data = self.normalize(primitive_data)
        return (primitive_data, *rest)
