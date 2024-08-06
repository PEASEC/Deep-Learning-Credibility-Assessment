from .NeuralNetWrapper import NeuralNetWrapper
from torch.utils.data import DataLoader
from typing import Tuple, List
from .metrics import ConfusionMatrix, get_confusion_matrix
import torch


class BinaryNeuralValidator(NeuralNetWrapper):
    def __init__(self, model: torch.nn.Module, device: torch.device = None, collate_fn=None):
        super(BinaryNeuralValidator, self).__init__(model, device, collate_fn)
        self.loss_fn = torch.nn.BCELoss()

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def validate_and_predict(
            self,
            data: tuple,
            normalize: bool = False,
            loss_fn=None
    ) -> Tuple[List[float], float, float, ConfusionMatrix]:
        """
        Validates the model with a given validation set.

        Returns:
            predictions: A list of predictions
            loss: Loss as defined by loss_fn
            mae: Mean Absolute error
            acc: Accuracy
            acc_custom: Custom Accuracy, that only considers 0.4<=x<=0.6 values
        """
        if loss_fn is None:
            loss_fn = self.loss_fn

        primitive_data, *additional_data, labels = data
        if normalize:
            primitive_data = self.normalize(primitive_data)

        features: list = [primitive_data, *additional_data]
        with torch.no_grad():
            model = self.model.eval()
            for i in range(len(features)):
                features[i] = features[i].to(self.device)
            labels = labels.to(self.device)

            predictions = model(*features)
            predictions = torch.reshape(predictions, (-1,))
            loss = loss_fn(predictions, labels)
            mae = torch.nn.L1Loss()(predictions, labels)

            return predictions.tolist(), loss.item(), mae.item(), get_confusion_matrix(predictions, labels)

    def validate(self, data: tuple, normalize: bool = False, loss_fn=None) -> Tuple[float, float, ConfusionMatrix]:
        """
        Validates the model with a given validation set.

        Returns:
            loss: Loss as defined by loss_fn
            mae: Mean Absolute error
            acc: Accuracy
            acc_custom: Custom Accuracy, that only considers 0.4<=x<=0.6 values
        """
        _, loss, mae, matrix = self.validate_and_predict(data, normalize, loss_fn)
        return loss, mae, matrix

    def validate_and_predict_dataset(
            self,
            validation_set: torch.utils.data.Dataset,
            normalize: bool = False,
            loss_fn=None
    ) -> Tuple[List[float], float, float, ConfusionMatrix]:
        """
        Validates the model with a given validation set.

        Returns:
            predictions: A list of predictions
            loss: Loss as defined by loss_fn
            mae: Mean Absolute error
            acc: Accuracy
            acc_custom: Custom Accuracy, that only considers 0.4<=x<=0.6 values
        """
        data = self.load_all_from_dataset(validation_set)
        return self.validate_and_predict(data, normalize, loss_fn)

    def validate_dataset(
            self,
            validation_set: torch.utils.data.Dataset,
            normalize: bool = False,
            loss_fn=None
    ) -> Tuple[float, float, ConfusionMatrix]:
        """
        Validates the model with a given validation set.

        Returns:
            loss: Loss as defined by loss_fn
            mae: Mean Absolute error
            acc: Accuracy
            acc_custom: Custom Accuracy, that only considers 0.4<=x<=0.6 values
        """
        _, loss, mae, matrix = self.validate_and_predict_dataset(validation_set, normalize, loss_fn)
        return loss, mae, matrix
