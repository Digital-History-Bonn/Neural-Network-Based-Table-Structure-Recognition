# type: ignore
"""script to create tikz plots from tensorboard from https://github.com/Digital-History-Bonn/HistorischeTabellenSemanticExtraction/blob/main/scripts/tensorboard2tikz.py .

since tikzplotlib is not maintained anymore, this will only work with webcolors<=1.13
"""
from csv import reader
from pathlib import Path
from typing import List

import matplotlib.figure
import requests
import matplot2tikz as tikzplotlib
from matplotlib import pyplot as plt


def tikzplotlib_fix_ncols(obj):
    """Workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib from https://github.com/nschloe/tikzplotlib/issues/557.

    Args:
        obj: matplotlib fig

    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


def get(run: str, tag: str, metric: str) -> List[float]:
    """Gets data from tensorboard.

    Args:
        run: name of the run in tensorboard
        tag: 'Valid' or 'Train'
        metric: metric to get

    Returns:
        List of values
    """
    response = requests.get(
        f"http://localhost:6006/experiment/defaultExperimentId/data/plugin/scalars/scalars?tag={tag}%2F{metric}&run={run}&format=csv"
    )
    data = response.text
    data_csv = reader(data.splitlines())
    values = [float(x[2]) for x in list(data_csv)[1:]]

    return values


def plot(runs: List[str], tag: str, metric: str):
    """Plots data from tensorboard of runs given in list.

    Args:
        runs: name of the run in tensorboard
        tag: 'Valid' or 'Train'
        metric: metric to get

    """
    data = [get(r, tag, metric) for r in runs]

    fig = plt.figure()
    for values, run in zip(data, runs):
        plt.plot(values, label=run)

    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.title(f"{tag}-{metric}")
    savepath = f"{Path(__file__).parent.absolute()}/plots/{tag}-{metric}.tex"
    save_plot_as_tikz(fig, savepath)
    plt.show()


def save_plot_as_tikz(fig: matplotlib.figure, savepath: str):
    """Save matplotlib figure as tikz plot.

    Args:
        fig: matplotlib figure
        savepath: save location

    """
    assert savepath.endswith(".tex")
    tikzplotlib_fix_ncols(fig)
    tikzplotlib.save(savepath)


if __name__ == "__main__":
    pass
