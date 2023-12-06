"""
Copyright 2022 Clare Lyle, University of Oxford
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
from typing import AnyStr, List
from slingpy import AbstractDataSource
from slingpy.models.abstract_base_model import AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import (
    BaseBatchAcquisitionFunction,
)
from discobax.methods.bax_acquisition import bax_sampling_toy_experiment

top_k_idx = bax_sampling_toy_experiment.top_k_idx


class ThompsonSamplingcquisition(BaseBatchAcquisitionFunction):
    def __call__(
        self,
        dataset_x: AbstractDataSource,
        select_size: int,
        available_indices: List[AnyStr],
        cumulative_indices: List[AnyStr] = None,
        model: AbstractBaseModel = None,
        dataset_y: AbstractDataSource = None,
        alpha=lambda t: 10.0,
        t=1,
    ) -> List:
        if type(dataset_x) == np.ndarray:
            avail_dataset_x = dataset_x[available_indices]
        else:
            avail_dataset_x = dataset_x.subset(available_indices)

        model_pedictions = model.predict(avail_dataset_x, return_std_and_margin=True)

        pred_mean, pred_uncertainties = model_pedictions[:2]
        og_save_dir = model.save_path_dir
        model.save_path_dir = "/tmp/model"
        model.save()
        n = len(dataset_x)
        avail_sample = []
        new_x = model.X
        new_y = model.mu
        avail_sample = np.array([])
        for x in avail_dataset_x:
            model.load()
            f = model.sample(x.reshape(1, -1))
            if len(new_x.shape) == 2:
                new_x = np.concatenate([new_x, x.reshape(1, -1)])
            else:
                new_x = np.concatenate(
                    [
                        new_x,
                        x.reshape(
                            1,
                        ),
                    ]
                )
            new_y = np.concatenate(
                [
                    new_y,
                    f.reshape(
                        1,
                    ),
                ]
            )
            model.fit(new_x, new_y)
            avail_sample = np.concatenate(
                [
                    avail_sample,
                    f.reshape(
                        1,
                    ),
                ]
            )
        best_indices = top_k_idx(np.asarray(avail_sample).reshape(-1), select_size)
        proposal = np.asarray(available_indices)[best_indices]
        model.load()
        model.save_path_dir = og_save_dir
        return proposal
