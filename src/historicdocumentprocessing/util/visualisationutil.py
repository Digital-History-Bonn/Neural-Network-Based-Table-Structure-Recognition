import json
import os
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes

from src.historicdocumentprocessing.tabletransformer_dataset import (
    reversetablerelativebboxes_outer_rowcoll,
)
from src.historicdocumentprocessing.util.tablesutil import remove_invalid_bbox, reversetablerelativebboxes_outer, \
    extractboxes


def drawimg_varformat_inner(box, impath, savepath, groundpath=None, rowcol=False):
    # print(impath)
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
                # gbox = reversetablerelativebboxes_outer_rowcoll(groundpath, "row")
                torch.vstack(
                    (
                        reversetablerelativebboxes_outer_rowcoll(groundpath, "row"),
                        reversetablerelativebboxes_outer_rowcoll(groundpath, "col"),
                    )
                )
        labels.extend(["Ground"] * gbox.shape[0])
        colors.extend(["red"] * gbox.shape[0])
        # print(gbox)
        box = torch.vstack((box, gbox))
    # print(labels, colors, box)
    # print(gbox.shape, box.shape)
    # print(img.shape)
    # print(predpath)
    # print(gbox)
    # newbox = remove_invalid_bbox(box, impath)
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
    # plt.imshow(image.permute(1, 2, 0))
    if savepath:
        os.makedirs(savepath, exist_ok=True)
        identifier = impath.split("/")[-1].split(".")[-2]
        # plt.savefig(f"{savepath}/{identifier}")
        # cv2.imwrite(img=image.numpy(), filename=f"{savepath}/{identifier}.jpg")
        img = Image.fromarray(image.permute(1, 2, 0).numpy())
        # print(f"{savepath}/{identifier}.jpg")
        img.save(f"{savepath}/{identifier}.jpg")


def drawimg(
    impath: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData/test/Konflikttabelle.jpg",
    jsonpath: str = f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Konflikttabelle.jpg.json",
    savepath: str = None,
):
    # img = plt.imread(impath)
    # img.setflags(write=1)
    # img = img.copy()
    # print(img.flags)
    img = read_image(impath)
    with open(jsonpath) as s:
        box = extractboxes(json.load(s))
    # print(img.shape)
    image = draw_bounding_boxes(image=img, boxes=box)
    plt.imshow(image.permute(1, 2, 0))
    if savepath:
        os.makedirs(savepath, exist_ok=True)
        identifier = impath.split("/")[-1].split(".")[-2]
        # plt.savefig(f"{savepath}/{identifier}")
        # cv2.imwrite(img=image.numpy(), filename=f"{savepath}/{identifier}.jpg")
        img = Image.fromarray(image.permute(1, 2, 0).numpy())
        img.save(f"{savepath}/{identifier}.jpg")
    plt.show()
