"""Utility functions."""

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.utils import draw_bounding_boxes


def get_image(image: torch.Tensor, boxes: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Draws bounding boxes on the given image and returns it as torch Tensor.

    Args:
        image: image to draw bounding boxes on
        boxes: Dict of bounding boxes to draw on the image.
               Keys are used as labels

    Returns:
            torch.Tensor of image with bounding boxes

    Raises:
        ValueError: if image tensor doesn't have 3 (color) channel in dim 0 and 2
    """
    image = image.clone()
    # unbatch image if image has batch dim
    image = image[0] if image.dim() == 4 else image
    # move color channel first if color channel is last
    image = image.permute(2, 0, 1) if image.shape[2] == 3 else image
    # if first dim doesn't have 3 raise Error
    if image.shape[0] != 3:
        raise ValueError("Only RGB image, need to have 3 channels in dim 0 or 2")

    # map [0, 1] to [0, 255]
    if image.max() <= 1.0:
        image *= 256

    colorplate = ["green", "red", "blue", "yellow"]

    colors = []
    labels = []
    coords = torch.zeros((0, 4))

    # change order to draw predictions on top of ground truth
    boxes_lst = list(boxes.items())
    boxes_lst.reverse()
    for idx, (label, item) in enumerate(boxes_lst):
        labels.extend([label] * len(item))
        colors.extend([colorplate[idx]] * len(item))
        coords = torch.vstack((coords, item))

    result = draw_bounding_boxes(image.to(torch.uint8), coords, colors=colors, width=3)

    return result


def plot_image(
    image: Union[torch.Tensor, np.ndarray],  # type: ignore
    boxes: Dict[str, torch.Tensor],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot an image with given bounding boxes using pyplot.

    Args:
        image: image to plot
        boxes: dictionary with bounding boxes. Keys are labels.
        title: title of the plot
        save_path: path to save the plot as image
    """
    if isinstance(image, np.ndarray):
        image = torch.tensor(image)

    # create image with annotation
    image = get_image(image, boxes)

    # plot image
    plt.imshow(image.permute(1, 2, 0))

    # add title if existing
    if title is not None:
        plt.title(title)

    # save images if save_as folder is given
    if save_path is not None:
        path = f"{Path(__file__).parent.absolute()}/../../../data/assets/images/{save_path}/"
        os.makedirs(path, exist_ok=True)
        plt.savefig(f"{path}{save_path}_{title}.png")

    # show plot
    plt.show()


def plot_annotations(folder: str, save_as: Optional[str] = None) -> None:
    """Shows target for model from given folder.

    Args:
        folder: path to folder of preprocessed example
        save_as: name of the folder to save the images in (optional)
                 Images are saved in data/assets/images/
    """
    image = [img for img in glob.glob(f"{folder}/*.jpg") if "table" not in img][0]
    tables = sorted(glob.glob(f"{folder}/*_table_*.jpg"), key=lambda x: int(x[-5]))
    tableregions = list(glob.glob(f"{folder}/*_tables.pt"))
    textregions = list(glob.glob(f"{folder}/*_textregions.pt"))
    cells_list = sorted(glob.glob(f"{folder}/*_cell_*.pt"), key=lambda x: int(x[-4]))
    cols_list = sorted(glob.glob(f"{folder}/*_col_*.pt"), key=lambda x: int(x[-4]))
    rows_list = sorted(glob.glob(f"{folder}/*_row_*.pt"), key=lambda x: int(x[-4]))

    pass

    # plot tables
    if tableregions:
        print(image)
        plot_image(
            plt.imread(image),
            {"tables": torch.load(tableregions[0])},
            title="tables",
            save_path=save_as,
        )

    # plot tables and textregions if textregions exists
    if tableregions and textregions:
        plot_image(
            plt.imread(image),
            boxes={
                "tables": torch.load(tableregions[0]),
                "textregions": torch.load(textregions[0]),
            },
            title="tables and textregions",
            save_path=save_as,
        )

    # plot cells, rows and columns
    for img, cells, rows, cols in zip(tables, cells_list, rows_list, cols_list):
        plot_image(
            plt.imread(img),
            boxes={"cells": torch.load(cells)},
            title="cells",
            save_path=save_as,
        )
        plot_image(
            plt.imread(img),
            boxes={"rows": torch.load(rows)},
            title="rows",
            save_path=save_as,
        )
        plot_image(
            plt.imread(img),
            boxes={"columns": torch.load(cols)},
            title="columns",
            save_path=save_as,
        )


def convert_coords(string: str) -> np.ndarray:  # type: ignore
    """Takes a string of coordinates from a xml file and converts it to a numpy array.

    Args:
        string: string of coordinates

    Returns:
        np.ndarray with coordinates

    """
    points = [[int(num) for num in x.split(",")] for x in string.split()]

    return np.array(points)


def get_bbox(
    points: np.ndarray,  # type: ignore
    corners: Union[None, List[int]] = None,
    tablebbox: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[int, int, int, int]:
    """Creates a bounding box around all given points.

    Args:
        points: np.ndarray of shape (N x 2) containing a list of points
        corners: corners can be defined if this is the case only the corner points are used for bb
        tablebbox: if given, bbox is calculated relative to table

    Returns:
        coordinates of bounding box in the format (x_min, y_min, x_max, y_max)

    """
    if corners:
        points = points[corners]

    x_max, x_min = points[:, 0].max(), points[:, 0].min()
    y_max, y_min = points[:, 1].max(), points[:, 1].min()
    # swap 0 and 1 for tablebox if np.flip in extract_glosat_annotation
    if tablebbox:
        x_min -= tablebbox[0]
        y_min -= tablebbox[1]
        x_max -= tablebbox[0]
        y_max -= tablebbox[1]

    return x_min, y_min, x_max, y_max


if __name__ == "__main__":
    plot_annotations(
        f"{Path(__file__).parent.absolute()}/../../../data/GloSAT/train/0", save_as="0"
    )
