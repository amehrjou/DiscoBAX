"""
Copyright 2022 Arash Mehrjou, GlaxoSmithKline plc
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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, AnyStr


def sine1d(inputs: np.array) -> np.array:
    outputs = np.sin(inputs)
    return outputs


def sinc1d(inputs: np.array) -> np.array:
    outputs = np.sin(inputs) / inputs
    return outputs


def sine2d(inputs: np.array) -> np.array:
    outputs = np.sin(inputs)
    return outputs


def sinc2d(inputs: np.array) -> np.array:
    """Computes the sinc(x)=sin(x)/x

    Args:
        inputs (np.array): (nsamples x dims) numpy array from the function domain

    Returns:
        outputs (np.array):  (nsamples x 1) numpy array
    """
    outputs = outputs = np.sin(np.linalg.norm(inputs, ord=2)) / inputs
    return outputs
