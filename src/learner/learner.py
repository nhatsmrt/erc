import torch
from torch.nn import Module
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from nntoolbox.callbacks import Callback, CallbackHandler
from nntoolbox.metrics import Metric
from nntoolbox.utils import get_device, load_model
from nntoolbox.transforms import MixupTransformer
from nntoolbox.learner import Learner
from typing import Iterable, Dict


__all__ = ['SupervisedSequenceLearner']


class SupervisedSequenceLearner(Learner):
    def __init__(
            self, train_data: DataLoader, val_data: DataLoader, model: Module,
            criterion: Module, optimizer: Optimizer, device=get_device(), mixup: bool = False, mixup_alpha: float = 0.4
    ):
        super().__init__(train_data, val_data, model, criterion, optimizer)
        self._device = device
        self._mixup = mixup
        if mixup:
            self._mixup_transformer = MixupTransformer(alpha=mixup_alpha)

    def learn(
            self,
            n_epoch: int, callbacks: Iterable[Callback] = None,
            metrics: Dict[str, Metric] = None, final_metric: str = 'accuracy', load_path=None
    ) -> float:
        if load_path is not None:
            load_model(self._model, load_path)

        self._cb_handler = CallbackHandler(self, n_epoch, callbacks, metrics, final_metric)
        self._cb_handler.on_train_begin()

        for e in range(n_epoch):
            print("Epoch " + str(e))
            self._model.train()
            self._cb_handler.on_epoch_begin()

            for inputs, lengths, labels in self._train_data:
                self.learn_one_iter(inputs, lengths, labels)

            stop_training = self.evaluate()
            if stop_training:
                print("Patience exceeded. Training finished.")
                break

        return self._cb_handler.on_train_end()

    def learn_one_iter(self, inputs: Tensor, lengths: Tensor, labels: Tensor):
        data = self._cb_handler.on_batch_begin({'inputs': inputs, 'lengths': lengths, 'labels': labels}, True)
        inputs = data['inputs']
        lengths = data['lengths']
        labels = data['labels']

        if self._mixup:
            inputs, labels = self._mixup_transformer.transform_data(inputs, labels)

        self._optimizer.zero_grad()
        loss = self.compute_loss(inputs, lengths, labels, True)
        loss.backward()
        self._optimizer.step()
        if self._device.type == 'cuda':
            mem = torch.cuda.memory_allocated(self._device)
            self._cb_handler.on_batch_end({"loss": loss.cpu(), "allocated_memory": mem})
        else:
            self._cb_handler.on_batch_end({"loss": loss})

    @torch.no_grad()
    def evaluate(self) -> float:
        self._model.eval()
        all_outputs = []
        all_labels = []
        total_data = 0
        loss = 0

        for inputs, lengths, labels in self._val_data:
            data = self._cb_handler.on_batch_begin({'inputs': inputs, 'lengths': lengths, 'labels': labels}, False)
            inputs = data['inputs']
            lengths = data['lengths']
            labels = data['labels']

            all_outputs.append(self.compute_outputs(inputs, lengths, False))
            all_labels.append(labels)
            loss += self.compute_loss(inputs, lengths, labels, False).cpu().item() * len(inputs)
            total_data += len(inputs)

        loss /= total_data

        logs = dict()
        logs["loss"] = loss
        logs["outputs"] = torch.cat(all_outputs, dim=0)
        logs["labels"] = torch.cat(all_labels, dim=0)

        return self._cb_handler.on_epoch_end(logs)

    def compute_outputs(self, inputs: Tensor, lengths: Tensor, train: bool) -> Tensor:
        return self._cb_handler.after_outputs({"output": self._model(inputs, lengths)}, train)["output"]

    def compute_loss(self, inputs: Tensor, lengths: Tensor, labels: Tensor, train: bool) -> Tensor:
        if self._mixup:
            criterion = self._mixup_transformer.transform_loss(self._criterion, self._model.training)
        else:
            criterion = self._criterion
        outputs = self.compute_outputs(inputs, lengths, train)

        return self._cb_handler.after_losses({"loss": criterion(outputs, labels)}, train)["loss"]
