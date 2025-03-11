"""Visualisation utility functions."""
import json
import os

import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes

from src.historicdocumentprocessing.tabletransformer_dataset import (
    reversetablerelativebboxes_outer_rowcoll,
)
from src.historicdocumentprocessing.util.tablesutil import remove_invalid_bbox, reversetablerelativebboxes_outer, extractboxes
from typing import Optional


def drawimg_varformat_inner(box: torch.Tensor, impath: str, savepath: Optional[str] = None, groundpath: Optional[str] = None, rowcol: bool = False):
    """Inner function for drawing image with predictions.

    Args:
        box: bbox tensor
        impath: path to image
        savepath: where to save images
        groundpath: path to ground truth
        rowcol: wether its row/column or cell BBoxes

    """
    img = read_image(impath)
    labels = ["Pred" for i in range(box.shape[0])]
    colors = ["green" for i in range(box.shape[0])]
    if groundpath:
        if "json" in groundpath:
            with open(groundpath) as s:
                gbox = extractboxes(json.load(s))
        elif "pt" in groundpath:
            gbox = torch.load(groundpath)
        else:
            gbox = reversetablerelativebboxes_outer(groundpath)
            if rowcol:
                gbox = torch.vstack(
                    (
                        reversetablerelativebboxes_outer_rowcoll(groundpath, "row"),
                        reversetablerelativebboxes_outer_rowcoll(groundpath, "col"),
                    )
                )
        labels.extend(["Ground"] * gbox.shape[0])
        colors.extend(["red"] * gbox.shape[0])
        box = torch.vstack((box, gbox))
    try:
        image = draw_bounding_boxes(image=img, boxes=box, labels=labels, colors=colors)
    except ValueError:
        # print(predpath)
        # print(box)
        newbox = remove_invalid_bbox(box, impath)
        labels = ["Pred" for i in range(newbox.shape[0])]
        image = draw_bounding_boxes(
            image=img, boxes=newbox, labels=labels, colors=colors
        )
    if savepath:
        os.makedirs(savepath, exist_ok=True)
        identifier = impath.split("/")[-1].split(".")[-2]
        img = Image.fromarray(image.permute(1, 2, 0).numpy())
        img.save(f"{savepath}/{identifier}.jpg")


def drawimg(
    impath: str,  # f"{Path(__file__).parent.absolute()}/../../data/BonnData/test/Konflikttabelle.jpg"
    jsonpath: str,  # f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Konflikttabelle.jpg.json"
    savepath: Optional[str] = None,
):
    """Function to draw bboxes from json file on image.

    Args:
        impath: path to image
        jsonpath: path to json file
        savepath: where to save image

    """
    img = read_image(impath)
    with open(jsonpath) as s:
        box = extractboxes(json.load(s))
    image = draw_bounding_boxes(image=img, boxes=box)
    plt.imshow(image.permute(1, 2, 0))
    if savepath:
        os.makedirs(savepath, exist_ok=True)
        identifier = impath.split("/")[-1].split(".")[-2]
        img = Image.fromarray(image.permute(1, 2, 0).numpy())
        img.save(f"{savepath}/{identifier}.jpg")
    plt.show()
