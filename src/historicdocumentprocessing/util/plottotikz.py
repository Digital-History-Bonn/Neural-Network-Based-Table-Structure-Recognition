"""
script to create tikz plots from tensorboard
from https://github.com/Digital-History-Bonn/HistorischeTabellenSemanticExtraction/blob/main/scripts/tensorboard2tikz.py
"""

from pathlib import Path
from typing import List

import requests
from csv import reader

import tikzplotlib
from matplotlib import pyplot as plt


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    from https://github.com/nschloe/tikzplotlib/issues/557
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


def get(run: str, tag: str, metric: str) -> List[float]:
    """
    Gets data from tensorboard
    :param run: name of the run in tensorboard
    :param tag: 'Valid' or 'Train'
    :param metric: metric to get

    :return: List of values
    """
    response = requests.get(f'http://localhost:6006/experiment/defaultExperimentId/data/plugin/scalars/scalars?tag={tag}%2F{metric}&run={run}&format=csv')
    data = response.text
    data_csv = reader(data.splitlines())
    values = [float(x[2]) for x in list(data_csv)[1:]]

    return values


def plot(runs: List[str], tag: str, metric: str):
    """
    plots data from tensorboard of runs given in list
    :param runs: name of the run in tensorboard
    :param tag: 'Valid' or 'Train'
    :param metric: metric to get

    :return: List of values
    """

    data = [get(r, tag, metric) for r in runs]

    fig = plt.figure()
    for values, run in zip(data, runs):
        plt.plot(values, label=run)

    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel(metric)
    plt.title(f"{tag}-{metric}")
    tikzplotlib_fix_ncols(fig)
    tikzplotlib.save(f'{Path(__file__).parent.absolute()}/plots/{tag}-{metric}.tex')
    plt.show()


if __name__ == '__main__':
    plot(['run_tables1', 'run_cols1'], 'Valid', 'loss')