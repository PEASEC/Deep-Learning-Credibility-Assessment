import torch
import os
from torch.utils.data import Dataset
import numpy as np
from src.util import SuspendManager, get_repository_path
from src.ml import BinaryNeuralLearner, LearningHistory, MappingDataset, split_from_dict, \
    random_split_percentage, generate_classification_histogram, get_confusion_matrix
from src.twitter import TweetDataset, PostObject, LabeledPostObject
from typing import Tuple, Any, List, cast, Sequence, Generic, TypeVar
from enum import Enum
from src.features import FeatureExtractor
from tqdm import tqdm
from timeit import default_timer as timer

torch.manual_seed(42)
np.random.seed(42)


class FeatureTypes(Enum):
    BASIS_ONLY = ""
    TEXT_ONLY = "text"
    TWEET_ONLY = "tweet"
    USER_ONLY = "user"
    TIMELINE = "timeline"
    TEXT_TWEET = "text-tweet"
    TWEET_USER = "tweet-user"
    TEXT_USER = "text-user"
    TEXT_TWEET_USER = "text-tweet-user"
    TEXT_TIMELINE_TWEET_USER = "text-timeline-tweet-user"
    TIMELINE_TWEET_USER = "timeline-tweet-user"
    TIMELINE_USER = "timeline-user"


T = TypeVar('T')


class HyperParameterSettings:
    def __init__(
            self,
            learning_rate: float,
            batch_size: int,
            num_epochs: int,
            optimizer_fn,
            weight_decay: float,
            evaluating_frequency: int
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.evaluating_frequency = evaluating_frequency
        self.weight_decay = weight_decay

        if isinstance(optimizer_fn, str):
            if optimizer_fn.lower() == "adam":
                self.optimizer_fn = torch.optim.Adam
            elif optimizer_fn.lower() == "adamw":
                self.optimizer_fn = torch.optim.AdamW
            elif optimizer_fn.lower() == "adadelta":
                self.optimizer_fn = torch.optim.Adadelta
            elif optimizer_fn.lower() == "sgd":
                self.optimizer_fn = torch.optim.SGD
            else:
                raise ValueError("Unknown optimizer function: '{}'.".format(optimizer_fn))
        else:
            self.optimizer_fn = optimizer_fn


class Trainer:
    def __init__(self,
                 learner: BinaryNeuralLearner,
                 train_set: Dataset,
                 dev_set: Dataset,
                 test_set: Dataset
                 ):
        self.learner = learner
        self.train_set = train_set
        self.dev_set = dev_set
        self.test_set = test_set

    def report_result(self, history_obj: LearningHistory) -> str:
        result = ""
        result += "      | Train                | Dev                  | Test " + "\n"
        result += "Epoch | Loss   MAE    Acc    | Loss   MAE    Acc    | Loss   MAE    Acc" + "\n"

        for i, (acc, epoch, state) in enumerate(history_obj.get_best_values()):
            self.learner.load_state_dict(state)
            train_loss, train_abs_diff, train_matrix = self.learner.validate_dataset(self.train_set)
            dev_loss, dev_abs_diff, dev_matrix = self.learner.validate_dataset(self.dev_set)
            test_loss, test_abs_diff, test_matrix = self.learner.validate_dataset(self.test_set)

            result += "{:>5} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f}".format(
                epoch,
                train_loss, train_abs_diff, train_matrix.accuracy,
                dev_loss, dev_abs_diff, dev_matrix.accuracy,
                test_loss, test_abs_diff, test_matrix.accuracy,
            ) + "\n"

        return result

    def print_results(self, history_obj: LearningHistory):
        print()
        print(self.report_result(history_obj))

    def save_results(self, path: str, history_obj: LearningHistory):
        report = self.report_result(history_obj)
        with open(os.path.join(path, "results.txt"), "w") as file:
            file.write(report)

        (acc, epoch, state) = history_obj.get_best_values()[0]
        self.learner.load_state_dict(state)
        self.learner.save(os.path.join(path, "model.pt"))
        history_obj.save(os.path.join(path, "learning_history.csv"))

    def train(self, hyper_parameter: HyperParameterSettings) -> LearningHistory:
        history_obj = self.learner.train(
            train_set=self.train_set,
            dev_set=self.dev_set,
            learning_rate=hyper_parameter.learning_rate,
            batch_size=hyper_parameter.batch_size,
            num_epochs=hyper_parameter.num_epochs,
            optimizer_fn=hyper_parameter.optimizer_fn,
            weight_decay=hyper_parameter.weight_decay,
            evaluating_frequency=hyper_parameter.evaluating_frequency
        )

        return history_obj


class Executable:
    def __init__(
            self,
            model_path: str,
            feature_type: FeatureTypes = None,
            display_progress: bool = True,
            variant: str = "",
            device_name: str = None,
    ):
        self.model_path = model_path
        self.feature_type = feature_type
        self.name = self._get_name()
        self.display_progress = display_progress
        self.variant = variant

        if device_name is not None and device_name != "":
            self.device_name = device_name
        else:
            self.device_name = "cuda" if torch.cuda.is_available() else "cpu"

        # Cache
        self.feature_extractor = None
        self.model_wrapper = None

        self.subdirectory = ""

    @staticmethod
    def _has_label(dataset: Sequence[PostObject]):
        return len(dataset) > 0 and isinstance(dataset[0], LabeledPostObject)

    @staticmethod
    def _get_data_size(data_tuple: Tuple[Any, ...]) -> Tuple[torch.Size]:
        result: List[torch.Size] = []
        for data in data_tuple:
            if isinstance(data, torch.Tensor):
                result.append(data.size())
            elif isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], torch.Tensor):
                    result.append(data[0].size())
                else:
                    result.append(torch.Size())
            else:
                result.append(torch.Size())
        return tuple(result)

    @staticmethod
    def _compare_data_size(data_1: Tuple[torch.Size, ...], data_2: Tuple[torch.Size, ...]) -> bool:
        if len(data_1) != len(data_2):
            return False

        for data in zip(data_1, data_2):
            if len(data[0]) != len(data[1]):
                return False

            if len(data[0]) > 1:
                # First dimension is batch, we can ignore it
                if data[0][1:] != data[1][1:]:
                    return False

        return True

    def _get_name(self) -> str:
        pass

    def _get_dataset(self, dataset: Sequence[PostObject]) -> Tuple[Any, ...]:
        pass

    def _get_hyper_parameter(self) -> HyperParameterSettings:
        pass

    def _get_model(self, data_size: Tuple[torch.Size]) -> torch.nn.Module:
        pass

    # noinspection PyMethodMayBeStatic
    def _get_collate_fn(self):
        return None

    def get_result_path(self, feature_type: bool = True) -> str:
        base_path = os.path.join(self.model_path, self.name)
        if feature_type:
            feature_type = cast(str, self.feature_type.value)
            return os.path.join(base_path, feature_type, self.subdirectory)
        else:
            return base_path

    # noinspection PyMethodMayBeStatic
    def _progress_wrapper(self, iterable: Generic[T], desc: str = None) -> T:
        if self.display_progress:
            return tqdm(iterable, desc=desc)
        else:
            return iterable

    def has_text_features(self):
        return self.feature_type in [
            FeatureTypes.TEXT_ONLY,
            FeatureTypes.TEXT_USER,
            FeatureTypes.TEXT_TWEET,
            FeatureTypes.TEXT_TWEET_USER,
            FeatureTypes.TIMELINE_TWEET_USER
        ]

    def has_tweet_features(self):
        return self.feature_type in [
            FeatureTypes.TWEET_ONLY,
            FeatureTypes.TWEET_USER,
            FeatureTypes.TIMELINE_TWEET_USER,
            FeatureTypes.TEXT_TWEET,
            FeatureTypes.TEXT_TIMELINE_TWEET_USER,
            FeatureTypes.TEXT_TWEET_USER
        ]

    def has_user_features(self):
        return self.feature_type in [
            FeatureTypes.USER_ONLY,
            FeatureTypes.TIMELINE_TWEET_USER,
            FeatureTypes.TEXT_USER,
            FeatureTypes.TEXT_TWEET_USER,
            FeatureTypes.TWEET_USER,
            FeatureTypes.TEXT_TIMELINE_TWEET_USER,
            FeatureTypes.TIMELINE_USER
        ]

    def has_timeline_features(self):
        return self.feature_type in [
            FeatureTypes.TEXT_TIMELINE_TWEET_USER,
            FeatureTypes.TIMELINE_TWEET_USER,
            FeatureTypes.TIMELINE,
            FeatureTypes.TIMELINE_USER
        ]

    def  get_feature_extractor(self):
        if self.feature_extractor is not None:
            return self.feature_extractor

        self.feature_extractor = FeatureExtractor(
            text_features=self.has_text_features(),
            tweet_features=self.has_tweet_features(),
            user_features=self.has_user_features(),
            timeline_features=self.has_timeline_features(),
            timeline_uses_all_features=(self.feature_type == FeatureTypes.TIMELINE)
        )

        return self.feature_extractor

    def get_model_wrapper(self, data_size: Tuple[torch.Size], path: str = None) -> BinaryNeuralLearner:
        if self.model_wrapper is not None and self._compare_data_size(self.model_wrapper[0], data_size):
            return self.model_wrapper[1]
        else:
            model_wrapper = BinaryNeuralLearner(self._get_model(data_size), device=torch.device(self.device_name))
            model_wrapper.set_collate_fn(self._get_collate_fn())
            model_wrapper.set_loss_fn(torch.nn.MSELoss())

            if path is not None:
                model_wrapper.load(path)

            self.model_wrapper = (data_size, model_wrapper)
            return model_wrapper

    def get_model_to_learn(self, data_size: Tuple[torch.Size]):
        return self.get_model_wrapper(data_size)

    def train(self, tweets: TweetDataset, ignore_cache: bool = True):
        os.makedirs(self.get_result_path(), exist_ok=True)

        raw_data: Tuple[Any, ...]
        feature_tmp_path = os.path.join(self.get_result_path(), "features.pt")
        if os.path.exists(feature_tmp_path) and not ignore_cache:
            print("Loading cached features")
            raw_data = torch.load(feature_tmp_path)
        else:
            raw_data = self._get_dataset(tweets)
            torch.save(raw_data, feature_tmp_path)

        raw_parameter = self._get_hyper_parameter()
        learner = self.get_model_to_learn(self._get_data_size(raw_data))

        # First input of data is a simple vector, which needs to be normalized
        x_features, *rest = raw_data
        x_features = learner.normalize_train(x_features)
        dataset = MappingDataset(x_features, *rest)

        split = tweets.splitting_information
        train_set, dev_set, test_set = \
            split_from_dict(dataset, split) if split is not None else random_split_percentage(dataset, [0.8, 0.1, 0.1])

        trainer = Trainer(learner, train_set, dev_set, test_set)
        # noinspection PyUnusedLocal
        history_obj = None
        with SuspendManager("Learner", "Training classifier"):
            history_obj = trainer.train(raw_parameter)

        print("Evaluating results...")
        trainer.print_results(history_obj)

        print("Saving results...")
        trainer.save_results(self.get_result_path(), history_obj)

        return history_obj

    def validate(self, tweets: TweetDataset, predictions_save_file: str = None):
        raw_data = self._get_dataset(tweets)
        validator = self.get_model_wrapper(
            self._get_data_size(raw_data),
            get_repository_path(os.path.join(self.get_result_path(), "model.pt"))
        )

        raw_data = validator.normalize_tuple(raw_data)
        dataset = MappingDataset(*raw_data)

        predictions, loss, mae, matrix = validator.validate_and_predict_dataset(dataset, normalize=False)

        print("{:>15}: {:.4f}".format("Loss", loss))
        print("{:>15}: {:.4f}".format("MAE", mae))
        print("{:>15}: {:.4f}".format("Accuracy", matrix.accuracy))
        print("{:>15}: {:.4f}".format("Precision", matrix.precision))
        print("{:>15}: {:.4f}".format("Recall", matrix.recall))
        print("{:>15}: {:.4f}".format("F1-Score", matrix.f1_score))
        print("{:>15}: {:.4f} (inverse)".format("F1-Score", matrix.inverted().f1_score))
        print("{:>15}: {:.4f} (macro)".format("F1-Score", (matrix.f1_score + matrix.inverted().f1_score) / 2))

        print()
        print(matrix)

        print()
        print("Binary borders for accuracy")
        borders = torch.FloatTensor(range(31, 70)) / 100
        prediction_tensor = torch.FloatTensor(predictions)
        accuracy_by_border = [
            get_confusion_matrix(prediction_tensor, raw_data[-1], binary_border=border).accuracy for border in borders
        ]

        for i in range(0, len(borders), 10):
            placeholder = "{:>5.4f}  " * (min(len(borders) - i, 10) - 1) + "{:>5.4f}"
            end = min(len(borders), i + 10)
            print("Border:   " + placeholder.format(*borders[i:end]))
            print("Accuracy: " + placeholder.format(*accuracy_by_border[i:end]))
            print()



        if predictions_save_file is not None and predictions_save_file != "":
            print("Saving predictions...")
            with open(predictions_save_file, "w") as file:
                file.write("tweet_id,source,label,prediction\n")
                for ((tweet, (source, label)), prediction) in zip(tweets.meta_data.items(), predictions):
                    file.write("{},{},{},{}\n".format(tweet, source, label, prediction))

        print()
        histogram = generate_classification_histogram(predictions, raw_data[-1])
        print("{:>6} {:>9} {:>9} {:>9}".format("Bucket", "Instances", "Correct", "Incorrect"))
        for entry in histogram:
            print("{:>6.1f} {:>9} {:>9} {:>9}".format(
                entry.lower_bucket,
                entry.instances,
                entry.correctly_classified,
                entry.incorrectly_classified)
            )

        # plt.scatter(raw_data[-1].tolist(), predictions)
        # plt.title("Predictions and Ground Truth")
        # plt.xlabel("Ground Truth")
        # plt.ylabel("Predictions")
        # plt.show()

        return loss, mae, matrix

    def predict(self, dataset: Sequence[PostObject]):
        raw_data = self._get_dataset(dataset)
        wrapper = self.get_model_wrapper(
            self._get_data_size(raw_data),
            get_repository_path(os.path.join(self.get_result_path(), "model.pt"))
        )

        raw_data = wrapper.normalize_tuple(raw_data)
        dataset = MappingDataset(*raw_data)
        return wrapper.predict_dataset(dataset, normalize=False)

    def predict_and_measure_time(self, dataset: Sequence[PostObject]):
        print("Loading dataset to cache")
        tweets: List[PostObject] = [
            t.remove_label() if isinstance(t, LabeledPostObject) else t for t in dataset
        ]
        self.display_progress = False

        print("Initializing dataset loader")
        dataset_initializing_before_time = timer()
        self._get_dataset([])
        dataset_initializing_after_time = timer()

        tmp_raw_data = self._get_dataset(tweets[:3])

        print("Load model file")
        model_load_before_time = timer()
        self.get_model_wrapper(
            self._get_data_size(tmp_raw_data),
            get_repository_path(os.path.join(self.get_result_path(), "model.pt"))
        )
        model_load_after_time = timer()

        self.display_progress = True
        print("Generate dataset")
        dataset_creation_before_time = timer()
        raw_data = self._get_dataset(tweets)
        dataset_creation_after_time = timer()

        # Wrapper is already loaded
        wrapper = self.model_wrapper[1]
        raw_data = wrapper.normalize_tuple(raw_data)
        dataset = MappingDataset(*raw_data)

        print("Executing model")
        predictions_creation_before_time = timer()
        predictions = wrapper.predict_dataset(dataset, normalize=False)
        predictions_creation_after_time = timer()

        format_str = "{:>10}: {:>8.4f}s"
        print()
        print("===== Initialization =====")
        dataset_initializing_time = dataset_initializing_after_time - dataset_initializing_before_time
        model_load_time = model_load_after_time - model_load_before_time
        print(format_str.format("Features", dataset_initializing_time))
        print(format_str.format("Model", model_load_time))
        print(format_str.format("(Total)", dataset_initializing_time + model_load_time))

        e_format_str = format_str + " ({:.8f}ms / Entry)"
        print("======= Executing ========")
        dataset_creation_time = dataset_creation_after_time - dataset_creation_before_time
        predictions_creation_time = predictions_creation_after_time - predictions_creation_before_time
        print(e_format_str.format("Features", dataset_creation_time, dataset_creation_time / len(tweets) * 1000))
        print(e_format_str.format("Model", predictions_creation_time, predictions_creation_time / len(tweets) * 1000))
        print(e_format_str.format(
            "(Total)",
            dataset_creation_time + predictions_creation_time,
            (dataset_creation_time + predictions_creation_time) / len(tweets) * 1000
        ))
        print("--------------------------")
        print(format_str.format(
            "(Total)", dataset_creation_time + predictions_creation_time + dataset_initializing_time + model_load_time
        ))

        return predictions

    def _get_executor_copy(self) -> "Executable":
        return self.__class__(
            self.model_path,
            self.feature_type,
            self.display_progress,
            self.variant,
            self.device_name
        )

    def copy(self, feature_type: FeatureTypes = None):
        obj = self._get_executor_copy()
        if feature_type is None or feature_type == self.feature_type:
            obj.model_wrapper = self.model_wrapper
            obj.feature_extractor = self.feature_extractor
        else:
            obj.feature_type = feature_type

        return obj

    def __copy__(self):
        return self.copy()
