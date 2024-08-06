import math
import torch
import sys
from torch.utils.data import Dataset, random_split, Subset, DataLoader
from typing import List, Dict
from src.util import read_simple_config_file


def random_split_percentage(dataset: Dataset, percentages: List[float]) -> List[Subset]:
    if sum(percentages) == 1:
        percentages = percentages[:-1]
    elif sum(percentages) > 1:
        raise ValueError("Sum of all percentages is greater than one.")

    total_length = len(dataset)
    lengths = [math.floor(p * total_length) for p in percentages]
    lengths.append(total_length - sum(lengths))
    return random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))


def split_from_dict(dataset: Dataset, splitting: Dict[str, List[int]]) -> List[Subset]:
    return [Subset(dataset, splitting[key]) for key in splitting]


def split_from_file(dataset: Dataset, splitting_file_name: str) -> List[Subset]:
    content = read_simple_config_file(splitting_file_name)

    def convert(val: str):
        return [int(i.strip()) for i in val.split(",")]

    result = []
    if "train" in content:
        result.append(Subset(dataset, convert(content["train"])))

    if "dev" in content:
        result.append(Subset(dataset, convert(content["dev"])))

    if "test" in content:
        result.append(Subset(dataset, convert(content["test"])))

    return result


def load_all_from_dataset(data, collate_fn=None):
    d = DataLoader(
        data,
        batch_size=sys.maxsize,
        drop_last=False,
        shuffle=False,
        collate_fn=collate_fn
    )
    return next(iter(d))
