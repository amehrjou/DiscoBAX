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
Contains the utility functions for creating 1 and 2 dimensional MoG functions
and samples from these functions.
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, AnyStr, Sequence
import scipy.stats as ss
import functools


def mog_1d(
    n_samples: Optional[int] = 100,
    limits: Tuple[float, float] = [-1, 1],
    means: Optional[Sequence[float]] = [-0.5, 0.5],
    stds: Optional[Sequence[float]] = [0.1, 0.1],
    weights: Optional[Sequence[float]] = [0.3, 0.7],
):
    """Make a groundtruth MoG function with the provided limits as the domain."""

    def mog_pdf(
        means: Optional[Sequence[float]] = [-0.5, 0.5],
        stds: Optional[Sequence[float]] = [0.1, 0.1],
        weights: Optional[Sequence[float]] = [0.3, 0.7],
    ) -> Callable:
        component_pdfs = []
        n_components = len(means)
        for comp_index in range(n_components):
            component_pdfs.append(
                functools.partial(
                    ss.norm.pdf, loc=means[comp_index], scale=stds[comp_index]
                )
            )

        def pdf_func(x):
            out = 0
            for comp_index in range(n_components):
                out += weights[comp_index] * component_pdfs[comp_index](x)
            return out

        return pdf_func

    pdf = mog_pdf(means, stds, weights)
    x_samples = np.linspace(limits[0], limits[1], n_samples)
    y_samples = pdf(x_samples)

    return x_samples, y_samples, pdf
