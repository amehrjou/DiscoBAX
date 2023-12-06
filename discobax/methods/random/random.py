"""
Copyright 2022 Arash Mehrjou, GlaxoSmithKline plc, Clare Lyle, University of Oxford, Pascal Notin, University of Oxford
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
from slingpy.utils.logging import info
from typing import List, AnyStr
from slingpy import AbstractDataSource, AbstractBaseModel

class RandomBatchAcquisitionFunction(object):
    def __call__(
        self,
        dataset_x: AbstractDataSource,
        acquisition_batch_size: int,
        available_indices: List[AnyStr],
        last_selected_indices: List[AnyStr], 
        cumulative_indices: List[AnyStr] = None,
        model: AbstractBaseModel = None,
        dataset_y: AbstractDataSource = None,
        temp_folder_name: str = 'tmp/model',
    ):
        info(len(available_indices), acquisition_batch_size, " randomacquisition")
        selected = np.random.choice(available_indices, size=acquisition_batch_size, replace=False)
        return selected
