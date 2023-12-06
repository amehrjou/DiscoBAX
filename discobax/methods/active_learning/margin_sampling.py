import numpy as np
from typing import AnyStr, List
from slingpy import AbstractDataSource, AbstractBaseModel
from discobax.methods.active_learning.base_acquisition_function import (
    BaseBatchAcquisitionFunction,
)


class MarginSamplingAcquisition(BaseBatchAcquisitionFunction):
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
            raise TypeError("The provided model does not output margins.")

        pred_mean, pred_uncertainties, pred_margins = model_pedictions

        if len(pred_mean) < acquisition_batch_size:
            raise ValueError(
                "The number of query samples exceeds" "the size of the available data."
            )

        numerical_selected_indices = np.flip(np.argsort(pred_margins))[
            :acquisition_batch_size
        ]
        selected_indices = [available_indices[i] for i in numerical_selected_indices]

        return selected_indices
