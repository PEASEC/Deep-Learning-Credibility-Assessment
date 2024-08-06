import sys
from enum import Enum
from typing import List, Tuple, Callable, TypeVar, Generic, Iterator

T = TypeVar('T')


class OptimizerMethod(Enum):
    MIN = 0
    MAX = 1


class BestNElements(Generic[T]):
    def __init__(self, n, method: OptimizerMethod = OptimizerMethod.MAX):
        self.array: List[Tuple[float, T]] = []
        self.n = n

        if method == OptimizerMethod.MIN:
            self.max = False
            self.min = True
            for i in range(n):
                self.array.append((sys.maxsize, None))
        elif method == OptimizerMethod.MAX:
            self.max = True
            self.min = False
            for i in range(n):
                self.array.append((-sys.maxsize, None))
        else:
            raise ValueError("Type must be either min or max")

    def __get_index(self, metric) -> int:
        for i in range(self.n):
            curr_metric, curr_data = self.array[i]
            if (self.max and metric > curr_metric) or (self.min and metric < curr_metric):
                return i
        return -1

    def __truncate(self):
        if len(self.array) > self.n:
            self.array = self.array[:self.n]

    def update(self, metric: float, data: T):
        index = self.__get_index(metric)
        if index > -1:
            self.array.insert(index, (metric, data))

        self.__truncate()

    def update_on_request(self, metric: float, data_fn: Callable[[], T]):
        index = self.__get_index(metric)
        if index > -1:
            self.array.insert(index, (metric, data_fn()))

        self.__truncate()

    def to_list(self):
        return [(m, v) for (m, v) in self.array if abs(m) != sys.maxsize]

    def __len__(self):
        return len(self.to_list())

    def __iter__(self) -> Iterator[Tuple[float, T]]:
        return iter(self.to_list())

    def __getitem__(self, item):
        return self.to_list()[item]
