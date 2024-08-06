import torch
from torch.utils.data import Dataset, TensorDataset
from transformers import BertTokenizer
from src.ml.model.transformers import BertExtractor
from src.features import preprocess_bert
from src.ml.model.BertFullyConnected import NeuralNet
from .Executable import Executable, HyperParameterSettings, FeatureTypes
from typing import Tuple, Any
from src.twitter import TweetDataset


class ModelHyperParameter:
    def __init__(
            self,
            dropout: float,
            bert_drop_p: float,
            hidden_size: float
    ):
        self.dropout = dropout
        self.bert_drop_p = bert_drop_p
        self.hidden_size = hidden_size


class BertExecutable(Executable):
    def __init__(self,
                 model_path: str,
                 pretained_bert_path: str,
                 feature_type: FeatureTypes = None,
                 display_progress: bool = True,
                 variant: str = "",
                 device_name: str = None
                 ):
        super(BertExecutable, self).__init__(
            model_path, feature_type, display_progress, variant, device_name
        )

        self.bert_path = pretained_bert_path

        # Cache
        self.tokenizer = None
        self.extractor = None

    @staticmethod
    def get_name() -> str:
        return "bert"

    def _get_name(self) -> str:
        return BertExecutable.get_name()

    def _get_tokenizer(self) -> BertTokenizer:
        if self.tokenizer is None:
            # This was set to "bert-base-cased" during the master thesis
            # however the provided vocab file should be equal to bert-base-cased.
            self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
            self.tokenizer.add_tokens(["[unused1]", "[unused2]"], True)

        return self.tokenizer

    def _get_extractor(self) -> BertExtractor:
        if self.extractor is None:
            self.extractor = BertExtractor(self.bert_path, device_name=self.device_name)

        return self.extractor

    def _get_dataset(self, dataset: TweetDataset) -> Tuple[Any, ...]:
        x_features = []
        x_texts = []
        y_data = []
        has_label = self._has_label(dataset)
        f = self.get_feature_extractor()
        for obj in self._progress_wrapper(dataset, desc="Extract features"):
            features = f.get_feature_vector(obj)
            preprocess_bert("", url_token="", mention_token="", hashtag_token="", allcaps_token="", replace_emojis=True,
                            replace_smileys=False)
            x_texts.append(preprocess_bert(
                obj.post.text,
                url_token="[unused1]",
                mention_token="[unused2]",
                replace_emojis=True
            ))
            x_features.append(features)

            if has_label:
                y_data.append(obj.label)

        x_features = torch.FloatTensor(x_features)
        y_data = torch.FloatTensor(y_data)

        tokenizer = self._get_tokenizer()

        x_embeddings: Dataset
        if len(x_texts) > 0:
            raw_embeddings = tokenizer.batch_encode_plus(x_texts, truncation=True, padding=True, return_tensors='pt')
            x_embeddings = BertExtractor.convert_to_dataset(raw_embeddings)
        else:
            x_embeddings = TensorDataset(torch.empty(0))

        bert = self._get_extractor()
        bert_data = bert.extract(
            x_embeddings,
            progress_wrapper=lambda x: self._progress_wrapper(x, desc="Extracting last model Layer")
        )

        if has_label:
            return x_features, bert_data, y_data
        else:
            return x_features, bert_data

    def __hyper_parameter_for_default(self):
        parameter = HyperParameterSettings(
            learning_rate=0.0002,
            batch_size=64,
            num_epochs=5_000,
            optimizer_fn=torch.optim.Adam,
            evaluating_frequency=2,
            weight_decay=0.1
        )
        return parameter

    def __model_for_default(self):
        model_parameter: ModelHyperParameter
        if self.feature_type == FeatureTypes.BASIS_ONLY:
            model_parameter = ModelHyperParameter(
                dropout=0.3,
                bert_drop_p=0,
                hidden_size=128
            )
        else:
            model_parameter = ModelHyperParameter(
                dropout=0.3,
                bert_drop_p=0.1,
                hidden_size=128
            )

        return model_parameter

    def __hyper_parameter_for_large(self):
        settings = HyperParameterSettings(
            learning_rate=0.0002,
            batch_size=64,
            num_epochs=1000,
            optimizer_fn=torch.optim.Adam,
            evaluating_frequency=2,
            weight_decay=0.1
        )

        if self.has_text_features():
            settings.num_epochs = 2000

        return settings

    def __model_for_large(self):
        model_parameter: ModelHyperParameter
        if self.feature_type == FeatureTypes.BASIS_ONLY:
            model_parameter = ModelHyperParameter(
                dropout=0.3,
                bert_drop_p=0,
                hidden_size=128
            )
        else:
            model_parameter = ModelHyperParameter(
                dropout=0.5,
                bert_drop_p=0.3,
                hidden_size=128
            )

        return model_parameter

    def _get_hyper_parameter(self) -> HyperParameterSettings:
        if self.variant.endswith("large"):
            return self.__hyper_parameter_for_large()
        else:
            return self.__hyper_parameter_for_default()

    def _get_model(self, dataset: Tuple[Any, ...]) -> torch.nn.Module:
        model_parameter: ModelHyperParameter
        if self.variant.endswith("large"):
            model_parameter = self.__model_for_large()
        else:
            model_parameter = self.__model_for_default()

        return NeuralNet(
            in_features=dataset[0][1] + dataset[1][1],
            out_features=1,
            dropout=model_parameter.dropout,
            bert_drop_p=model_parameter.bert_drop_p,
            hidden_size=model_parameter.hidden_size
        )

    def _get_executor_copy(self):
        result: BertExecutable = BertExecutable(
            self.model_path,
            self.bert_path,
            self.feature_type,
            self.display_progress,
            self.variant,
            self.device_name
        )
        result.tokenizer = self.tokenizer
        result.extractor = self.extractor
        return result
