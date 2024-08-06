from .Executable import Executable, FeatureTypes
import torch
import os
from src.twitter import PostObject
from typing import List, ClassVar
from src.features import FeatureExtractor


def advanced_timeline_feature_wrapper(
        base_class: ClassVar[Executable],
        name: str = "advanced",
        inner_model: Executable = None
) -> ClassVar[Executable]:
    class WithTimeLineFeatures(base_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if not self.has_timeline_features():
                raise ValueError("Timeline features must be used in order to use advanced timeline features")

        def get_feature_extractor(self):
            if self.has_timeline_features:
                mycopy: Executable
                if inner_model is None:
                    mycopy = self.copy()
                    mycopy.feature_type = FeatureTypes.TEXT_TWEET_USER
                    mycopy.feature_extractor = None
                    mycopy.model_wrapper = None
                    mycopy.display_progress = False
                else:
                    mycopy = inner_model

                def get_timeline_feature(obj: PostObject) -> List[float]:
                    dataset = [
                        PostObject(t, obj.user_profile)
                        for t in obj.user_timeline_posts
                        if len(t.text) > 0
                    ]
                    if len(dataset) <= 0:
                        return [0] * 4

                    prediction = mycopy.predict(dataset)
                    prediction_min = torch.min(prediction).item()
                    prediction_max = torch.max(prediction).item()
                    prediction_mean = torch.mean(prediction).item()

                    prediction_std_raw = torch.std(prediction)
                    prediction_std = 0.0 if torch.isnan(prediction_std_raw) else prediction_std_raw.item()

                    return [prediction_min, prediction_max, prediction_mean, prediction_std]

                self.feature_extractor = FeatureExtractor(
                    text_features=self.has_text_features(),
                    tweet_features=self.has_tweet_features(),
                    user_features=self.has_user_features(),
                    timeline_features=self.has_timeline_features(),
                    timeline_generator=get_timeline_feature
                )

                return self.feature_extractor
            else:
                return super().get_feature_extractor()

        def get_result_path(self, feature_type: bool = True) -> str:
            if feature_type:
                return os.path.join(super().get_result_path(), name)
            else:
                return super().get_result_path(False)

    return WithTimeLineFeatures
