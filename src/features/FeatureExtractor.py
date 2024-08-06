import inspect

from src.features.Features import TweetFeatures, UserFeatures, TextFeatures
from src.twitter import PostObject, AbstractPost, Tweet
from typing import List, Tuple, Any, Callable
import numpy as np
import multiprocessing


class FeatureExtractor:
    def __init__(
            self,
            tweet_features: bool = True,
            user_features: bool = True,
            text_features: bool = True,
            timeline_features: bool = True,
            timeline_uses_all_features: bool = False,
            timeline_generator: Callable[[PostObject], List[float]] = None
    ):
        self.tweet_features = tweet_features
        self.user_features = user_features
        self.text_features = text_features
        self.timeline_features = timeline_features
        self.timeline_uses_all_features = timeline_uses_all_features
        self.timeline_generator = timeline_generator

        if tweet_features or timeline_uses_all_features:
            TweetFeatures.initialize()
        if user_features or timeline_uses_all_features:
            UserFeatures.initialize()
        if text_features or timeline_uses_all_features:
            TextFeatures.initialize()

        self.__calculate_feature_vector_size()

    def __calculate_feature_vector_size(self):
        t = Tweet.from_file("resources/dummy_tweet.json")
        dummy_object = PostObject(t, {}, [])
        tweet_f, user_f, text_f = self.__get_basis_tweet_features(
            t, self.tweet_features, self.user_features, self.text_features
        )

        descriptions = []

        def get_feature_len(enabled, class_obj, class_instance):
            if enabled:
                values = FeatureExtractor.__get_properties(class_obj, class_instance)
                scalars, description = FeatureExtractor.__flatten_feature_list(values, return_description=True)
                descriptions.extend(description)
                return len(scalars)
            else:
                return 0

        self.tweet_features_length = get_feature_len(self.tweet_features, TweetFeatures, tweet_f)
        self.user_features_length = get_feature_len(self.user_features, UserFeatures, user_f)
        self.text_features_length = get_feature_len(self.text_features, TextFeatures, text_f)

        if self.timeline_features:
            self.timeline_features_length = len(self.get_timeline_feature_vector(dummy_object))
        else:
            self.timeline_features_length = 0

        self.descriptions = descriptions

    @staticmethod
    def __get_property_names(class_obj: object):
        def is_prop(v):
            return isinstance(v, property)

        return [name for (name, _) in inspect.getmembers(class_obj, is_prop)]

    @staticmethod
    def __get_properties(class_obj: object, class_instance: object):
        return [(name, getattr(class_instance, name)) for name in FeatureExtractor.__get_property_names(class_obj)]

    @staticmethod
    def __flatten_feature_list(features: List[Tuple[str, Any]], return_description: bool = False):
        vector: List[float] = []
        description: List[str] = []

        for (key, value) in features:
            if isinstance(value, tuple) or isinstance(value, list):
                vector.extend(value)
                if return_description:
                    description.extend(["{}_{:02d}".format(key, i) for i in range(len(value))])
            else:
                vector.append(value)
                if return_description:
                    description.append(key)

        if return_description:
            return vector, description
        else:
            return vector

    @staticmethod
    def __get_basis_tweet_features(
            tweet: AbstractPost,
            tweet_features: bool,
            user_features: bool,
            text_features: bool
    ) -> Tuple[TweetFeatures, UserFeatures, TextFeatures]:
        tweet_f = None
        user_f = None
        text_f = None

        if tweet_features:
            tweet_f = TweetFeatures(tweet)

        if user_features:
            user_f = UserFeatures(tweet)

        if text_features:
            text_f = TextFeatures(tweet)

        return tweet_f, user_f, text_f

    def get_vector_size(self) -> int:
        return self.tweet_features_length + self.user_features_length + self.text_features_length \
               + self.timeline_features_length

    def get_vector_description(self):
        user_feature_names = []
        if self.timeline_features:
            if self.timeline_generator is None:
                length = self.text_features_length + self.tweet_features_length
                user_feature_names.extend(["{}_{:02d}".format("timeline_min", i) for i in range(length)])
                user_feature_names.extend(["{}_{:02d}".format("timeline_max", i) for i in range(length)])
                user_feature_names.extend(["{}_{:02d}".format("timeline_mean", i) for i in range(length)])
                user_feature_names.extend(["{}_{:02d}".format("timeline_std", i) for i in range(length)])
            else:
                length = self.timeline_features_length
                user_feature_names.extend(["{}_{:02d}".format("timeline", i) for i in range(length)])

        return self.descriptions + user_feature_names

    def get_basis_tweet_feature_values(
            self,
            tweet: AbstractPost,
            tweet_features_override: bool = None,
            user_features_override: bool = None,
            text_features_override: bool = None
    ) -> List[Tuple[str, Any]]:
        tweet_features = tweet_features_override if tweet_features_override is not None else self.tweet_features
        user_features = user_features_override if user_features_override is not None else self.user_features
        text_features = text_features_override if text_features_override is not None else self.text_features

        tweet_f, user_f, text_f = self.__get_basis_tweet_features(
            tweet, tweet_features, user_features, text_features
        )

        result = []
        if tweet_f is not None:
            result.extend(FeatureExtractor.__get_properties(TweetFeatures, tweet_f))

        if user_f is not None:
            result.extend(FeatureExtractor.__get_properties(UserFeatures, user_f))

        if text_f is not None:
            result.extend(FeatureExtractor.__get_properties(TextFeatures, text_f))

        return result

    def get_timeline_feature_vector(self, obj: PostObject) -> List[float]:
        if self.timeline_generator is not None:
            return self.timeline_generator(obj)

        if len(obj.user_timeline_posts) <= 0:
            length = self.tweet_features_length + self.text_features_length
            result = [0] * length
            return result * 4

        vectors = []
        for tweet in obj.user_timeline_posts:
            features: list
            if self.timeline_uses_all_features:
                features = self.get_basis_tweet_feature_values(
                    tweet,
                    tweet_features_override=True,
                    user_features_override=False,
                    text_features_override=True
                )
            else:
                features = self.get_basis_tweet_feature_values(
                    tweet,
                    user_features_override=False
                )

            vectors.append(self.__flatten_feature_list(features))

        vectors = np.array(vectors)
        timeline_min = np.min(vectors, axis=0)
        timeline_max = np.max(vectors, axis=0)
        timeline_mean = np.mean(vectors, axis=0)
        timeline_std = np.std(vectors, axis=0)
        return [
            *timeline_min,
            *timeline_max,
            *timeline_mean,
            *timeline_std
        ]

    def get_feature_vector(self, tweet_object: PostObject):
        features = self.get_basis_tweet_feature_values(tweet_object.post)
        scalars = self.__flatten_feature_list(features)

        if self.timeline_features:
            features = self.get_timeline_feature_vector(
                tweet_object
            )
            scalars.extend(features)

        return scalars
