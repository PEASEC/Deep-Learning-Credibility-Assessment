import torch
from typing import NamedTuple, List, Union


class ConfusionMatrix(NamedTuple):
    """a docstring"""
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def item_count(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.item_count

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn)

    @property
    def f1_score(self) -> float:
        return self.get_f_score(beta=1)

    def get_f_score(self, beta: int = 1):
        return (1 + beta ** 2) * (self.precision * self.recall) / ((beta ** 2 * self.precision) + self.recall)

    def inverted(self):
        return ConfusionMatrix(
            tp=self.tn,
            tn=self.tp,
            fp=self.fn,
            fn=self.fp
        )


def get_confusion_matrix(pred: torch.Tensor, actual: torch.Tensor, binary_border: float = 0.5) -> ConfusionMatrix:
    pred = (pred >= binary_border).int()
    actual = (actual >= binary_border).int()

    tp = ((pred == 1) * (actual == 1)).sum().item()
    fp = ((pred == 1) * (actual == 0)).sum().item()
    fn = ((pred == 0) * (actual == 1)).sum().item()
    tn = ((pred == 0) * (actual == 0)).sum().item()

    return ConfusionMatrix(tp=tp, fp=fp, fn=fn, tn=tn)


def accuracy(pred: torch.Tensor, actual: torch.Tensor) -> float:
    pred = torch.round(pred)
    actual = torch.round(actual)
    actual[actual > 1] = 1
    actual[actual < 0] = 0

    return (pred == actual).float().sum().item() / len(pred)


def custom_accuracy(pred: torch.Tensor, actual: torch.Tensor) -> float:
    pred = [x.item() for x in pred]
    actual = [x.item() for x in actual]

    if len(pred) != len(actual):
        raise Exception("Length of prediction array must match length of actual array")

    count = 0
    correct = 0
    for i in range(len(pred)):
        if actual[i] <= 0.4 or actual[i] >= 0.6:
            count += 1
            if pred[i] <= 0.4 or pred[i] >= 0.6:
                if round(pred[i]) == round(actual[i]):
                    correct += 1

    if count > 0:
        return float(correct) / count
    else:
        return 0


class HistogramEntry:
    lower_bucket: float
    upper_bucket: float
    instances: int = 0
    correctly_classified: int = 0

    def __init__(self, lower_bucket: float, upper_bucket: float):
        self.lower_bucket = lower_bucket
        self.upper_bucket = upper_bucket

    @property
    def incorrectly_classified(self) -> int:
        return self.instances - self.correctly_classified


def generate_classification_histogram(
        prediction_tensor: Union[torch.Tensor, List[float]],
        actual_tensor: Union[torch.Tensor, List[float]],
        min_value: int = 0,
        max_value: int = 1,
        bucket_count: int = 10
) -> List[HistogramEntry]:
    pred_list = prediction_tensor.tolist() if isinstance(prediction_tensor, torch.Tensor) else prediction_tensor
    actual_list = actual_tensor.tolist() if isinstance(actual_tensor, torch.Tensor) else actual_tensor

    bucket_len = (max_value - min_value) / bucket_count
    buckets = [
        HistogramEntry(min_value + bucket_len * i, min_value + bucket_len * (i + 1)) for i in range(bucket_count)
    ]

    for (pred, actual) in zip(pred_list, actual_list):
        bucket_index = int((pred - min_value) / bucket_len)
        buckets[bucket_index].instances += 1
        if round(pred) == round(actual):
            buckets[bucket_index].correctly_classified += 1

    return buckets
