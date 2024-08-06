import torch
from transformers import BertModel
from typing import Union, Callable, Iterable
from torch.utils.data import Dataset, TensorDataset, DataLoader


class Extractor:
    def __init__(self, model, device_name: str = None):
        if device_name is None or device_name == "":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"

        if device_name == "cpu":
            self.device_name = "cpu"
            self.model = model.cpu()
        elif device_name in ["cuda", "gpu"]:
            self.device_name = "cuda"
            self.model = model.cuda()
        else:
            self.device_name = device_name
            self.model = model

        self.device = torch.device(self.device_name)
        self.cpu_forced = 0
        self.retry_on_cuda_fail = (device_name != "cpu")
        self.cpu_device = torch.device("cpu")

    def _convert_to_dataset(self, data: dict) -> Dataset:
        pass

    def extract_batch(self, batch) -> torch.Tensor:
        pass

    def _check_forced_cpu(self):
        if self.retry_on_cuda_fail and self.cpu_forced > 0:
            self.cpu_forced -= 1
            if self.cpu_forced == 0:
                try:
                    print("Trying cuda again")
                    self.cuda()
                except RuntimeError:
                    pass

    def extract(
            self,
            data: torch.utils.data.Dataset,
            batch_size: int = 64,
            progress_wrapper: Callable[[Iterable], Iterable] = None
    ) -> torch.Tensor:
        bert_layers = []
        with torch.no_grad():
            loader = DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False)
            if progress_wrapper is not None:
                loader = progress_wrapper(loader)

            for batch in loader:
                self._check_forced_cpu()
                try:
                    layer = self.extract_batch(batch)
                except RuntimeError as err:
                    if self.device_name == "cuda":
                        self.cpu()
                        self.cpu_forced = 20
                        print("Error on cuda device, continue on cpu.")
                    else:
                        raise err

                layer = layer.to(self.cpu_device)
                bert_layers.extend(layer)

        if len(bert_layers) > 0:
            return torch.stack(bert_layers)
        else:
            return torch.empty(0)

    def cpu(self):
        self.model.cpu()
        self.device_name = "cpu"
        self.device = torch.device(self.device_name)

    def cuda(self):
        self.model.cuda()
        self.device_name = "cuda"
        self.device = torch.device(self.device_name)
        return self


class BertExtractor(Extractor):
    def __init__(self, model_path: str = None, device_name: str = None):
        model = BertModel.from_pretrained("bert-base-cased" if model_path is None else model_path)
        super().__init__(model, device_name)

    @staticmethod
    def convert_to_dataset(data: Union[torch.utils.data.Dataset, dict]) -> torch.utils.data.Dataset:
        if isinstance(input, torch.utils.data.Dataset):
            return data

        return TensorDataset(
            data["input_ids"],
            data["token_type_ids"],
            data["attention_mask"]
        )

    def extract_batch(self, batch) -> torch.Tensor:
        input_ids: torch.Tensor = batch[0].to(self.device)
        token_type_ids: torch.Tensor = batch[1].to(self.device)
        attention_mask: torch.Tensor = batch[2].to(self.device)
        last_hidden_state, pooled_output = self.model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        return pooled_output
