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
import itertools 
import pdb


def cartesian_product(arrays):
    la = len(arrays)
    print(la)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def product_itertools(arrays):
    return np.array(list(itertools.product(*arrays)))

def sine_2d(
    n_samples: Optional[int] = 16**2,
    limits: Tuple[float, float] = [-2*np.pi, 2*np.pi],
):
    """Make a groundtruth sinusoidal function with the provided limits as the domain."""
    weights = np.asarray([[0.25, -1/np.pi], [0.1, .02]])
    
    def sin_fn(
        x, seed=0
    ) -> Callable:
        return np.mean(np.sin(weights @ x.T), axis=0)
    linspace = np.linspace(-np.pi, np.pi, int(n_samples**(1/2)))
    x_samples = product_itertools([linspace]*2)       
    y_samples = sin_fn(x_samples)

    return x_samples, y_samples, sin_fn

if __name__ == "__main__":
    xs, ys, f = sine_4d()
    n = int(np.sqrt(len(ys)))
    plt.plot( ys[:n])
    plt.plot(ys[::n])
    print(np.max(ys))
    plt.savefig("sin_2d.png")