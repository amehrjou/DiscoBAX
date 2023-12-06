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
from typing import List, AnyStr, Callable, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

_DEFAULT_DPI = 300
_X_LABEL_FONTSIZE = 20
_Y_LABEL_FONTSIZE = 20
_TICK_FONT_SIZE = 10
_TICK_GRID_LINEWIDTH = 20
_LEVEL_SET_LABEL_SIZE = 10


def plot_levelsets(
    func: Callable,
    samples: Optional[np.ndarray] = None,
    intervals: Tuple[Tuple[float, float], Tuple[float, float]] = ((-1, 1), (-1, 1)),
    margin: float = 0.0,
    labels: Tuple[AnyStr, AnyStr] = ["x", "y"],
    nlevels: int = 10,
    figure_size: Tuple[float, float] = (10, 10),
    res: int = 10,
    dpi: int = _DEFAULT_DPI,
    path_to_save=Optional[AnyStr],
):
    """take coordinate intervals and the height function and save
    the plot in full_path
    Args:
        interval: ((xmin, xmax), (ymin, ymax)), the limits of the axis to plot
        margin: A small value added to the plot limit
        full_path: path to save the image file
        res: resolution of the heatmap
        func: f([x, y]), a scalar-valued function whose levelsets are to be pllotted
        samples (n_samples, 2): If provided, overlay the scatter plot of samples on the heatmap.
        nlevels: The number of level sets (None for no level set)
        full_path: The full path to the saved plot. Do not save is None.
    """
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    xmin, xmax = intervals[0]
    ymin, ymax = intervals[1]
    xmin -= margin
    ymin -= margin
    xmax += margin
    ymax += margin
    x = np.linspace(xmin, xmax, res)
    y = np.linspace(ymin, ymax, res)
    xv, yv = np.meshgrid(x, y)
    xyv = np.vstack([xv.flatten(), yv.flatten()]).transpose()
    zv = []
    for p in xyv:
        zv.append(func(p))
    zv = np.array(zv).transpose()
    fig, ax = plt.subplots(figsize=figure_size, dpi=dpi, frameon=False)
    ax.pcolormesh(xv, yv, zv.reshape(xv.shape), cmap="viridis")
    if nlevels is not None:
        CS = ax.contour(xv, yv, zv.reshape(xv.shape), cmap="YlOrBr", levels=nlevels)
        ax.clabel(CS, inline=1, fontsize=_LEVEL_SET_LABEL_SIZE)
    if samples is not None:
        ax.plot(samples[:, 0], samples[:, 1], ".k")
    ax.set_xlabel(labels[0], fontsize=_X_LABEL_FONTSIZE)
    ax.set_ylabel(labels[1], fontsize=_Y_LABEL_FONTSIZE)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=_TICK_FONT_SIZE,
        grid_linewidth=_TICK_GRID_LINEWIDTH,
    )
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    if path_to_save is not None:
        plt.savefig(path_to_save, dpi=dpi)


def main():
    def f(x):
        return np.sqrt(x[0] ** 2 + x[1] ** 2)

    plot_levelsets(f, res=20, path_to_save="./test.png")


if __name__ == "__main__":
    main()
