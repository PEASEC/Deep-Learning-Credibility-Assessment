import torch
from src.ml.model.BaseNeuralNetwork import NeuralNet
from .Executable import Executable, HyperParameterSettings, FeatureTypes
from typing import Tuple, Any
from src.twitter import TweetDataset


class ModelHyperParameter:
    def __init__(
            self,
            dropout: float,
            hidden_size: int
    ):
        self.dropout = dropout
        self.hidden_size = hidden_size


class MlpExecutable(Executable):
    @staticmethod
    def get_name() -> str:
        return "mlp"

    def _get_name(self) -> str:
        return MlpExecutable.get_name()

    def _get_dataset(self, dataset: TweetDataset) -> Tuple[Any, ...]:
        x_features = []
        y_data = []

        has_label = self._has_label(dataset)
        f = self.get_feature_extractor()
        for obj in self._progress_wrapper(dataset, desc="Extract features"):
            features = f.get_feature_vector(obj)
            x_features.append(features)

            if has_label:
                y_data.append(obj.label)

        x_features = torch.FloatTensor(x_features)
        y_data = torch.FloatTensor(y_data)

        if has_label:
            return x_features, y_data
        else:
            return (x_features, )

    def _get_hyper_parameter(self) -> HyperParameterSettings:
        settings = HyperParameterSettings(
            learning_rate=0.01,
            batch_size=128,
            num_epochs=8000,
            optimizer_fn=torch.optim.Adam,
            evaluating_frequency=5,
            weight_decay=0
        )

        return settings

    def _get_model(self, data_size: Tuple[torch.Size]) -> torch.nn.Module:
        model_parameter: ModelHyperParameter
        if self.feature_type == FeatureTypes.BASIS_ONLY:
            raise ValueError("There is no basis type for the mlp model.")

        model_parameter = ModelHyperParameter(
            dropout=0.3,
            hidden_size=32
        )

        if self.variant.endswith("large"):
            model_parameter.dropout = 0.4

        # if self.feature_type == FeatureTypes.TEXT_TIMELINE_TWEET_USER:
        #     model_parameter.dropout = 0.4

        return NeuralNet(
            in_features=data_size[0][1],
            out_features=1,
            dropout=model_parameter.dropout,
            hidden_size=model_parameter.hidden_size
        )

    def _get_executor_copy(self):
        return MlpExecutable(self.model_path, self.feature_type, self.display_progress)
