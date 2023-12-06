import numpy as np
import scipy
from typing import AnyStr, List
from slingpy import AbstractDataSource, AbstractBaseModel
from discobax.methods.active_learning.base_acquisition_function import (
    BaseBatchAcquisitionFunction,
)


def softmax_temperature(x, temperature=1):
    """Computes softmax probabilities from unnormalized values

    Args:

        x: array-like list of energy values.
        temperature: a positive real value.

    Returns:
        outputs: ndarray or list (dependin on x type) that is
            exp(x / temperature) / sum(exp(x / temperature)).
    """
    if isinstance(x, list):
        y = np.array(x)
    else:
        y = x
    y = np.exp(y / temperature)
    out_np = scipy.special.softmax(y)
    if any(np.isnan(out_np)):
        raise ValueError("Temperature is too extreme.")
    if isinstance(x, list):
        return [out_item for out_item in out_np]
    else:
        return out_np


class TopUncertainAcquisition(BaseBatchAcquisitionFunction):
    def __call__(
        self,
        dataset_x: AbstractDataSource,
        acquisition_batch_size: int,
        available_indices: List[AnyStr],
        last_selected_indices: List[AnyStr],
        cumulative_indices: List[AnyStr] = None,
        model: AbstractBaseModel = None,
        dataset_y: AbstractDataSource = None,
        temp_folder_name: str = "tmp/model",
    ) -> List:
        avail_dataset_x = dataset_x.subset(available_indices)
        model_pedictions = model.predict(avail_dataset_x, return_std_and_margin=True)

        if len(model_pedictions) != 3:
            raise TypeError("The provided model does not output uncertainty.")

        pred_mean, pred_uncertainties, _ = model_pedictions

        if len(pred_mean) < acquisition_batch_size:
            raise ValueError(
                "The number of query samples exceeds" "the size of the available data."
            )

        numerical_selected_indices = np.flip(np.argsort(pred_uncertainties))[
            :acquisition_batch_size
        ]
        selected_indices = [available_indices[i] for i in numerical_selected_indices]

        return selected_indices


class SoftUncertainAcquisition(BaseBatchAcquisitionFunction):
    def __call__(
        self,
        dataset_x: AbstractDataSource,
        acquisition_batch_size: int,
        available_indices: List[AnyStr],
        last_selected_indices: List[AnyStr],
        cumulative_indices: List[AnyStr] = None,
        model: AbstractBaseModel = None,
        dataset_y: AbstractDataSource = None,
        temp_folder_name: str = "tmp/model",
        temperature: float = 0.5,
    ) -> List:
        avail_dataset_x = dataset_x.subset(available_indices)
        model_pedictions = model.predict(avail_dataset_x, return_std_and_margin=True)

        if len(model_pedictions) != 3:
            raise TypeError("The provided model does not output uncertainty.")

        pred_mean, pred_uncertainties, _ = model_pedictions

        if len(pred_mean) < acquisition_batch_size:
            raise ValueError(
                "The number of query samples exceeds" "the size of the available data."
            )
        selection_probabilities = softmax_temperature(
            np.log(1e-7 + pred_uncertainties**2),
            temperature,
        )
        try:
            numerical_selected_indices = np.random.choice(
                range(len(selection_probabilities)),
                size=acquisition_batch_size,
                replace=False,
                p=selection_probabilities,
            )
        except:
            numerical_selected_indices = range(acquisition_batch_size)
        selected_indices = [available_indices[i] for i in numerical_selected_indices]
        return selected_indices
