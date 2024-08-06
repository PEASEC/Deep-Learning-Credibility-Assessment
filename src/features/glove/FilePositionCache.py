"""
Provides a base class, that allows caching a certain amount of data.
After the capacity is reached, stored data is freed.
"""

from typing import Mapping, Generic, TypeVar, Dict, List

K = TypeVar("K")
V = TypeVar("V")


class FilePositionInfo:
    def __init__(self, position: int):
        self.position = position
        self.accesses = 0


class ValuedFilePositionInfo(Generic[K, V]):
    def __init__(self, info: FilePositionInfo, cache_position: int, key: K, value: V):
        self.info = info
        self.key = key
        self.value = value
        self.cache_position = cache_position

    def __repr__(self):
        return "<{}, accesses={}, cached_position={}>".format(self.key, self.info.accesses, self.cache_position)


class FilePositionCache(Mapping[K, int], Generic[K, V]):
    def __init__(self, capacity: int, store: Dict[K, FilePositionInfo] = None):
        if store is None:
            store = {}
        self.capacity = capacity
        self.store: Dict[K, FilePositionInfo] = store
        self.cache: Dict[K, ValuedFilePositionInfo[K, V]] = {}
        self.ordered_cache: List[ValuedFilePositionInfo[K, V]] = []

    def __find_position(self, accesses: int, start_position: int = -1):
        position = start_position
        while position > 0 and self.ordered_cache[position - 1].info.accesses < accesses:
            position -= 1
            self.ordered_cache[position].cache_position += 1
        return position

    def __increment_cached_index(self, obj: ValuedFilePositionInfo):
        obj.info.accesses += 1

        if obj.cache_position <= 0:
            return

        position = self.__find_position(obj.info.accesses, obj.cache_position)
        if position != obj.cache_position:
            # Position change
            del self.ordered_cache[obj.cache_position]
            self.ordered_cache.insert(position, obj)
            obj.cache_position = position

    def has_value(self, key: K) -> bool:
        return key in self.cache

    def get_value(self, key: K) -> V:
        obj = self.cache[key]
        self.__increment_cached_index(obj)
        return obj.value

    def set_value(self, key: K, value: V):
        if key in self.cache:
            return

        obj = self.store[key]
        obj.accesses += 1

        if len(self.cache) < self.capacity:
            position = len(self.ordered_cache)
            cache = ValuedFilePositionInfo(obj, position, key, value)
            self.ordered_cache.append(cache)
            self.cache[key] = cache
        elif obj.accesses > self.ordered_cache[-1].info.accesses:
            bad_item = self.ordered_cache.pop(-1)
            del self.cache[bad_item.key]

            position = self.__find_position(obj.accesses, len(self.ordered_cache))
            cache = ValuedFilePositionInfo(obj, position, key, value)
            self.ordered_cache.insert(position, cache)
            self.cache[key] = cache

    def get_cached_keys(self) -> List[K]:
        return [item.key for item in self.ordered_cache]

    def has_pointer(self, key: K) -> bool:
        return key in self.store

    def get_pointer(self, key: K) -> int:
        return self.store[key].position

    def __contains__(self, item):
        return item in self.store

    def keys(self):
        return self.store.keys()

    def values(self):
        return [entry.position for entry in self.store.values()]

    def get(self, k: K) -> int:
        if k in self.store:
            return self.store[k].position
        else:
            return -1

    def __eq__(self, other):
        if isinstance(other, FilePositionCache):
            return other.store == self.store
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getitem__(self, item) -> V:
        info = self.store[item]
        info.accesses += 1
        return info.position

    def __iter__(self):
        return self.store.__iter__()

    def __len__(self):
        return self.store.__len__()


class FilePositionCacheBuilder(Generic[K, V]):
    def __init__(self, capacity=1000):
        self.store: Dict[K, FilePositionInfo] = {}
        self.capacity = capacity

    def set_capacity(self, capacity) -> 'FilePositionCacheBuilder':
        self.capacity = capacity
        return self

    def add_item(self, key: K, position: int) -> 'FilePositionCacheBuilder':
        self.store[key] = FilePositionInfo(position)
        return self

    def build(self) -> FilePositionCache[K, V]:
        return FilePositionCache(self.capacity, self.store)
