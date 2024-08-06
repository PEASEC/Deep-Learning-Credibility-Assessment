from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import List, Tuple, Any
from src.util import BestNElements, OptimizerMethod
import torch
import matplotlib.pyplot as plt
import sys

import copy
import os

from .BinaryNeuralValidator import BinaryNeuralValidator


class LearningHistory:
    def __init__(self, metric_history: List[Tuple[float, ...]], best_results: List[Tuple[float, int, Any]] = None,
                 epoch_steps: int = 20, finished_with_error: bool = False):
        if best_results is None:
            best_results = []
        self.metric_history = torch.tensor(metric_history)
        self.best_results = best_results
        self.finished_with_error = finished_with_error
        self.epoch_steps = epoch_steps

    def plot(self):
        x_axis = range(0, len(self.metric_history) * self.epoch_steps, self.epoch_steps)

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.plot(x_axis, self.metric_history[:, 0])

        ax2 = ax1.twinx()
        ax2.set_ylabel("Accuracy (Dev)")
        ax2.plot(x_axis, self.metric_history[:, 3], color="tab:red")

        for (metric, index, _) in self.best_results:
            ax2.axvline(x=index, c="k")

        fig.tight_layout()
        plt.show()

    def get_metric_history(self):
        return self.metric_history

    def get_best_values(self):
        return self.best_results

    def save(self, file_name: str):
        with open(file_name, "w") as file:
            for obj in self.metric_history:
                file.write(",".join(str(x.item()) for x in obj))
                file.write("\n")


class BinaryNeuralLearner(BinaryNeuralValidator):
    def __save_tmp_model(self, epoch: int):
        os.makedirs("build/tmp", exist_ok=True)

        if epoch < 0:
            self.save("build/tmp/model_error.pt")
            self.save("build/tmp/model_error_{:0>5}.pt".format(epoch))
        else:
            self.save("build/tmp/{:0>5}.pt".format(epoch))

    def train(self,
              train_set: torch.utils.data.Dataset,
              learning_rate: float = 0.001,
              weight_decay: float = 0.1,
              batch_size: int = 16,
              num_epochs: int = 2000,
              epoch_offset: int = 0,
              evaluating_frequency: int = 20,
              loss_fn=None,
              optimizer_fn=None,
              dev_set: torch.utils.data.Dataset = None) -> LearningHistory:
        # Put model into training mode
        model = self.model.train()

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn
        )

        if loss_fn is None:
            loss_fn = self.loss_fn

        if optimizer_fn is None:
            optimizer_fn = torch.optim.Adadelta

        optimizer = optimizer_fn(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best: BestNElements[Tuple[int, Any]] = BestNElements(5, OptimizerMethod.MIN)
        history = []
        finished_with_error = False

        # # TMP
        # model.eval()
        # dev_loss, dev_mae, dev_acc, dev_custom_acc = self.validate_dataset(dev_set)
        # print("Acc: {}, Loss: {}".format(dev_acc, dev_loss))
        # exit()

        loop = tqdm(range(epoch_offset, num_epochs + epoch_offset), desc="Training")
        try:
            last_dev_loss = -1
            last_dev_accuracy = -1
            for epoch in loop:
                avg_loss = 0
                epoch_count = 0

                for batch_id, (*features, labels) in enumerate(train_loader):
                    # Note that in most cases features only contains one tensor of Dim
                    # (BxD) where B is the batch size and D is the feature size
                    # Sometimes a network needs additional data, like a sequence for
                    # the RNN. In this case, additional vectors are submitted
                    for i in range(len(features)):
                        features[i] = features[i].to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    predictions = model(*features)
                    predictions = torch.reshape(predictions, (-1,))

                    loss = loss_fn(predictions, labels)
                    avg_loss += loss.item()
                    epoch_count += 1

                    loss.backward()
                    optimizer.step()

                epoch_loss = avg_loss / float(epoch_count)
                if epoch % evaluating_frequency == 0:
                    if dev_set is None:
                        model.eval()
                        test_loss, test_mae, matrix = self.validate_dataset(train_set)
                        history.append((epoch_loss, test_loss, test_mae, matrix.accuracy))
                        model.train()
                        loop.set_postfix(loss=test_loss, accuracy=matrix.accuracy)
                    else:
                        model.eval()
                        dev_loss, dev_mae, matrix = self.validate_dataset(dev_set)
                        history.append((epoch_loss, dev_loss, dev_mae, matrix.accuracy))

                        best.update_on_request(dev_loss, lambda: (epoch, copy.deepcopy(model.state_dict())))
                        model.train()
                        last_dev_loss = dev_loss
                        last_dev_accuracy = matrix.accuracy

                loop.set_postfix(
                    dev_loss=last_dev_loss,
                    dev_accuracy=last_dev_accuracy,
                    train_loss=epoch_loss
                )

                if epoch % 1000 == 0 and epoch > epoch_offset:
                    self.__save_tmp_model(epoch)

        except Exception as error:
            print(error, file=sys.stderr)
            finished_with_error = True
            self.__save_tmp_model(-1)

        return LearningHistory(history, [(metric, index, state) for (metric, (index, state)) in best],
                               epoch_steps=evaluating_frequency, finished_with_error=finished_with_error)

    def normalize_train(self, data: torch.Tensor) -> torch.Tensor:
        # Calculate mean and std to normalize and zero center data
        self.mean = torch.mean(data, dim=0)
        self.std = torch.std(data, dim=0)

        # As we divide by std, we don't want it to be zero
        std_min_value = torch.empty(self.std.size()).fill_(0.001)
        self.std[(self.std < std_min_value)] = 1

        data -= self.mean
        data /= self.std

        return data
