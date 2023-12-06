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
import matplotlib.pyplot as plt

def sine_1d(
    n_samples: Optional[int] = 100,
    limits: Tuple[float, float] = [-2*np.pi, 2*np.pi],
):
    """Make a groundtruth sinusoidal function with the provided limits as the domain."""

    def sin_fn(
        x
    ) -> Callable:
        w1 = 2.5/np.pi 
        w2 = 2*np.pi/3
        w3 = 10
        return np.sin(w1 * x) + 0.95 * np.cos(w2 * x) + 0.25 * np.cos(w3 * x + 0.5)
    x_samples = np.linspace(limits[0], limits[1], n_samples)
    y_samples = sin_fn(x_samples)

    return x_samples, y_samples, sin_fn

if __name__ == "__main__":
    xs, ys, f = sine_1d()
    plt.plot(xs, ys)
    plt.savefig("sin.png")