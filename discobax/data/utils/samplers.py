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

******************
Contains the utility functions for creating samples from a given domain with
different distributions.
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, AnyStr, Sequence
import scipy.stats as ss
from sklearn.mixture import GaussianMixture


def mog_1d_sampler(
    n_samples: Optional[int] = 100,
    limits: Tuple[float, float] = [-1, 1],
    means: Optional[Sequence[float]] = [-0.5, 0.5],
    stds: Optional[Sequence[float]] = [0.2, 0.2],
    weights: Optional[Sequence[float]] = [0.3, 0.7],
):
    """Create samples from an MoG disribution from a specified domain."""

    gm = GaussianMixture(covariance_type="diag")
    n_components = len(means)
    gm.weights_ = np.array(weights)
    gm.means_ = np.array([-0.5, 0.5]).reshape((n_components, 1))
    gm.covariances_ = np.array(stds).reshape((n_components, 1)) ** 2
    samples, _ = gm.sample(n_samples)
    up_trancated_samples = np.where(samples < limits[1], samples, limits[1])
    down_trancated_samples = np.where(
        up_trancated_samples > limits[0], up_trancated_samples, limits[0]
    )
    return down_trancated_samples
