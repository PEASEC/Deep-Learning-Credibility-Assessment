import torch
import os
from src.ml.model import RNN
from src.twitter import PostObject
from src.features import load_embedding
from .Executable import Executable, HyperParameterSettings, FeatureTypes
from typing import Tuple, Any, Sequence
from src.util import get_repository_path

class ModelHyperParameter:
    def __init__(
            self,
            hidden_size: int,
            rnn_hidden_size: int,
            dropout_rate: float,
            rnn_dropout_rate: float,
            bidirectional: bool
    ):
        self.hidden_size = hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.dropout_rate = dropout_rate
        self.rnn_dropout_rate = rnn_dropout_rate
        self.bidirectional = bidirectional


class RnnExecutable(Executable):
    def __init__(self,
                 model_path: str,
                 feature_type: FeatureTypes = None,
                 display_progress: bool = True,
                 variant: str = "",
                 device_name: str = None,
                 keep_embeddings_in_memory: bool = False,
                 embedding_file: str = None
                 ):
        super(RnnExecutable, self).__init__(
            model_path, feature_type, display_progress, variant, device_name
        )

        self.keep_embeddings_in_memory = keep_embeddings_in_memory

        if embedding_file is None:
            embedding_file = get_repository_path("resources/embeddings/glove.twitter.27B.50d.enriched.txt")

        self.embedding_file = embedding_file

        # Cache
        self.embeddings = None

    @staticmethod
    def get_name():
        return "rnn"

    def _get_name(self) -> str:
        return RnnExecutable.get_name()

    def get_embeddings(self):
        if self.embeddings is not None:
            return self.embeddings
        else:
            print("Load Embeddings")
            self.embeddings = load_embedding(
                self.embedding_file,
                keep_in_memory=self.keep_embeddings_in_memory
            )
            return self.embeddings

    def _get_dataset(self, dataset: Sequence[PostObject]) -> Tuple[Any, ...]:
        has_label = self._has_label(dataset)

        glove = self.get_embeddings()

        x_data = []
        x_data_embeddings = []
        y_data = []

        f = self.get_feature_extractor()
        # Ensure embedding file stays lock -> faster access
        with glove:
            for obj in self._progress_wrapper(dataset, desc="Extract features"):
                features = f.get_feature_vector(obj)
                x_data_embeddings.append(glove.get_tweet_embeddings(obj.post.text))

                x_data.append(features)
                if has_label:
                    y_data.append(obj.label)

        x_data = torch.FloatTensor(x_data)
        y_data = torch.FloatTensor(y_data)

        if has_label:
            return x_data, x_data_embeddings, y_data
        else:
            return x_data, x_data_embeddings

    def __hyper_parameter_for_default(self):
        return HyperParameterSettings(
            learning_rate=0.01,
            weight_decay=0.002,
            batch_size=256,
            num_epochs=7_000,
            optimizer_fn=torch.optim.Adadelta,
            evaluating_frequency=5
        )

    def __model_for_default(self):
        return ModelHyperParameter(
            hidden_size=32,
            rnn_hidden_size=40,
            dropout_rate=0.5,
            rnn_dropout_rate=0,
            bidirectional=False
        )

    def __hyper_parameter_for_large(self):
        return HyperParameterSettings(
            learning_rate=0.02, # 0.01
            weight_decay=0.002,  # 0.002 !important
            batch_size=128,  # 64
            num_epochs=3_000,  # 3000
            optimizer_fn=torch.optim.Adadelta,
            evaluating_frequency=5
        )

    def __model_for_large(self):
        return ModelHyperParameter(
            hidden_size=128,
            rnn_hidden_size=64,
            dropout_rate=0.4,
            rnn_dropout_rate=0,
            bidirectional=False
        )

    def _get_hyper_parameter(self) -> HyperParameterSettings:
        learning_parameters = \
            self.__hyper_parameter_for_large() if self.variant.endswith("large") else self.__hyper_parameter_for_default()

        if self.feature_type != FeatureTypes.BASIS_ONLY:
            # We preload the rnn, so we do not need lots of epochs
            learning_parameters.num_epochs = 1000
        if self.feature_type in [FeatureTypes.TEXT_TWEET_USER, FeatureTypes.TWEET_USER, FeatureTypes.TIMELINE_TWEET_USER]:
            # More parameters need more regularization
            learning_parameters.weight_decay = 0.003
            learning_parameters.num_epochs = 1000

        if self.has_timeline_features():
            learning_parameters.num_epochs = 1800

        return learning_parameters

    def _get_model(self, tuple_size: Tuple[torch.Size]) -> torch.nn.Module:
        model_parameter = \
            self.__model_for_large() if self.variant.endswith("large") else self.__model_for_default()

        model = RNN(
            tuple_size[0][1],
            1,
            hidden_size=model_parameter.hidden_size,
            rnn_hidden_size=model_parameter.rnn_hidden_size,
            embedding_size=tuple_size[1][1],
            dropout_rate=model_parameter.dropout_rate,
            rnn_dropout_rate=model_parameter.rnn_dropout_rate,
            bidirectional=model_parameter.bidirectional
        )

        return model

    def get_model_to_learn(self, data_size: Tuple[torch.Size]):
        learner = self.get_model_wrapper(data_size)
        if self.feature_type != FeatureTypes.BASIS_ONLY:
            path = os.path.join(
                self.get_result_path(feature_type=False), "model.pt"
            )
            path = get_repository_path(path)

            if not os.path.exists(path):
                raise ValueError("No basis model available. You need to create a basis model first.")

            print("Load model")
            learner.model.load_rnn_state(path)

        return learner



    def _get_collate_fn(self):
        return RNN.collate_fn

    def _get_executor_copy(self):
        result: RnnExecutable = RnnExecutable(
            self.model_path,
            self.feature_type,
            self.display_progress,
            self.variant,
            self.device_name,
            self.keep_embeddings_in_memory,
            self.embedding_file
        )
        result.embeddings = self.embeddings
        return result
