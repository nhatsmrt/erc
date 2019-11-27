import torch
from torch import Tensor, nn
from torch.utils.data import Dataset
import numpy as np
from numpy import ndarray
import pandas as pd


__all__ = ['EmoRec']


class DummyTransform:
    def __call__(self, input): return input


class EmoRec:
    """
    Abstraction for an image classifier. Support user defined test time augmentation
    """
    def __init__(self, model, transform_main=DummyTransform(), tta_transform=None, tta_beta: float=0.4):
        self._model = model
        self._model.eval()
        self._tta_transform = tta_transform
        self._softmax = nn.Softmax(dim=1)
        self.transform_main = transform_main
        self.tta_beta = tta_beta

    @torch.no_grad()
    def predict(self, image: Tensor, tries: int=5) -> ndarray:
        """
        Predict the classes or class probabilities of a batch of images

        :param image: image to be predicted
        :param tries: number of tries for augmentation
        :return:
        """
        if self._tta_transform is not None:
            version = [self.transform_main(image)] + [self._tta_transform(image) for _ in range(tries)]
            outputs = self._softmax(self._model(torch.stack(version, dim=0)))
            weights = [self.tta_beta] + [(1 - self.tta_beta) / tries for _ in range(tries)]
            weights = torch.from_numpy(np.array(weights)).to(outputs.dtype).to(outputs.device)[:, None]
            outputs = (outputs * weights).sum(dim=0).argmax(0, keepdim=True)
        else:
            outputs = self._model(self.transform_main(image).unsqueeze(0)).argmax(1)

        return outputs.detach().cpu().numpy()

    @torch.no_grad()
    def export_predictions(self, test_data: Dataset, path: str, tries: int=5):
        outputs = []
        for image in test_data:
            outputs.append(self.predict(image, tries=tries))

        outputs = np.concatenate(outputs, axis=0)
        df_submission = pd.DataFrame({"File": test_data.filenames, "Label": outputs})
        df_submission.to_csv(path, index=False)
