import os
import json
import csv
from .PostObject import LabeledPostObject
from .Tweet import Tweet
from typing import List, Tuple, Dict, Any, Iterator, Union, overload, Sequence
from src.util import read_simple_config_file


def get_json_object(file):
    with open(file) as file_ptr:
        return json.load(file_ptr)


def load_dataset_meta(path, filter_by_src: list = None):
    result: Dict[str, Tuple[str, float]] = {}
    with open(path) as file:
        reader = csv.reader(file)
        for index, [tweet_id, source, label] in enumerate(reader):
            if index > 0:
                if filter_by_src is None or source in filter_by_src:
                    result[tweet_id] = (source, float(label))
    return result


def load_splitting_ids(path) -> Dict[str, List[int]]:
    config = read_simple_config_file(path)

    def to_int_list(val: str):
        return [int(i.strip()) for i in val.split(",")]

    result = {}
    for set_type in ["train", "dev", "test"]:
        if set_type in config:
            result[set_type] = to_int_list(config[set_type])

    return result


class TweetDataset(Sequence[LabeledPostObject]):
    def __init__(self, path, load_user_profile=True, load_user_timeline=True, filter_by_src: list = None,
                 user_timeline_max_items: int = -1):
        self.path = path
        self.load_user_profile = load_user_profile
        self.load_user_timeline = load_user_timeline
        self.user_timeline_max_items = user_timeline_max_items

        self.tweet_folder = os.path.join(path, "tweets")
        self.user_profile_folder = os.path.join(path, "user_profiles")
        self.user_timeline_folder = os.path.join(path, "user_timeline_tweets")
        self.meta_data = load_dataset_meta(os.path.join(path, "tweets.csv"), filter_by_src)
        self.tweet_id_list = list(self.meta_data.keys())

        splitting_file = os.path.join(path, "splitting.txt")
        if os.path.exists(splitting_file):
            self.splitting_information = load_splitting_ids(splitting_file)
        else:
            self.splitting_information = None

    def __get_tweet_object(self, twitter_id: str) -> LabeledPostObject:
        tweet = get_json_object(os.path.join(self.tweet_folder, twitter_id + ".json"))
        user_id = tweet["user"]["id"]
        user_profile: Union[None, Any] = None
        user_timeline: Union[None, Any] = None
        user_profile_path = "{}/{}.json".format(self.user_profile_folder, user_id)
        user_timeline_path = "{}/{}.json".format(self.user_timeline_folder, user_id)

        if self.load_user_profile and os.path.exists(user_profile_path):
            user_profile = get_json_object(user_profile_path)

        if self.load_user_timeline and os.path.exists(user_timeline_path):
            user_timeline: List[Any] = get_json_object(user_timeline_path)

            if self.user_timeline_max_items >= 0:
                user_timeline = user_timeline[:self.user_timeline_max_items]

        (source, label) = self.meta_data[tweet["id_str"]]

        tweet_obj = Tweet(tweet)
        timeline_obj = [Tweet(p) for p in user_timeline] if user_timeline is not None else None

        return LabeledPostObject(tweet_obj, label, source, user_profile, timeline_obj)

    @property
    def is_splittable(self):
        return self.splitting_information is not None

    def split(self) -> Tuple['TweetDataset', ...]:
        if not self.is_splittable:
            raise ValueError("Dataset cannot be split")

        ordered = ["train", "dev", "test"]
        result: List[TweetDataset] = []
        for set_type in ordered:
            if set_type in self.splitting_information:
                ids: List[int] = self.splitting_information[set_type]
                result.append(self[ids])

        return tuple(result)

    def __len__(self):
        return len(self.tweet_id_list)

    def __contains__(self, item):
        if hasattr(item, "tweet") and hasattr(item.activity, "id"):
            return item.activity.id in self.tweet_id_list
        else:
            return False

    def __eq__(self, other):
        if not isinstance(other, TweetDataset):
            return False

        return self.tweet_folder == other.tweet_folder

    def __iter__(self) -> Iterator[LabeledPostObject]:
        for tweet_id in self.tweet_id_list:
            yield self.__get_tweet_object(tweet_id)

    def __reversed__(self) -> Iterator[LabeledPostObject]:
        for tweet_id in reversed(self.tweet_id_list):
            yield self.__get_tweet_object(tweet_id)

    def __copy__(self):
        copy = TweetDataset(path=self.path, load_user_profile=self.load_user_profile,
                            load_user_timeline=self.load_user_timeline,
                            user_timeline_max_items=self.user_timeline_max_items)
        copy.meta_data = self.meta_data
        copy.tweet_id_list = self.tweet_id_list
        return copy

    def count(self, x: Any) -> int:
        if x in self:
            return 1
        else:
            return 0

    def index(self, x: Any, start: int = ..., end: int = ...) -> int:
        if isinstance(x, str):
            return self.tweet_id_list.index(x, start, end)

        if hasattr(x, "tweet"):
            x = x.activity

        if hasattr(x, "id"):
            return self.tweet_id_list.index(x.id, start, end)

    @overload
    def __getitem__(self, value: str) -> LabeledPostObject:
        ...

    @overload
    def __getitem__(self, value: slice) -> 'TweetDataset':
        ...

    @overload
    def __getitem__(self, value: list) -> 'TweetDataset':
        ...

    @overload
    def __getitem__(self, value: int) -> LabeledPostObject:
        ...

    def __copy_meta_data(self, include_keys: List[str]):
        result: Dict[str, Tuple[str, float]] = {}
        for key in include_keys:
            result[key] = self.meta_data[key]

        return result

    def __getitem__(self, value):
        if isinstance(value, str):
            return self.__get_tweet_object(value)
        elif isinstance(value, int):
            return self.__get_tweet_object(self.tweet_id_list[value])
        elif isinstance(value, slice):
            copy = self.__copy__()
            copy.tweet_id_list = self.tweet_id_list[value]
            copy.meta_data = self.__copy_meta_data(copy.tweet_id_list)
            copy.splitting_information = None
            return copy
        elif isinstance(value, list):
            copy = self.__copy__()
            copy.tweet_id_list = [self.tweet_id_list[i] for i in value]
            copy.meta_data = self.__copy_meta_data(copy.tweet_id_list)
            copy.splitting_information = None
            return copy
        else:
            raise ValueError("Unsupported index type: " + str(type(value)))
