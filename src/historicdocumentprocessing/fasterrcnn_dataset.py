"""Dataset Class for training. Code  modified from
https://github.com/Digital-History-Bonn/HistorischeTabellenSemanticExtraction/blob/main/src/TableExtraction
/customdataset.py"""

import glob
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
from PIL import Image
from torch.nn import Module
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

from src.historicdocumentprocessing.kosmos_eval import reversetablerelativebboxes_outer


class CustomDataset(Dataset):  # type: ignore
    """Dataset Class for training."""

    def __init__(
        self, path: str, objective: str, transforms: Optional[Module] = None
    ) -> None:
        """
        Dataset Class for training.

        Args:
            path: path to folder with images
            objective: detection type (fullimage only for now)
            transforms: torchvision transforms for on-the-fly augmentations
        """
        super().__init__()
        if objective == "fullimage":
            self.data = sorted(
                list(glob.glob(f"{path}/*")), key=lambda x: str(x.split(os.sep)[-1])
            )
        else:
            "not yet implemented since there is no need, left in so it can be added in future"
            pass
        self.objective = objective
        self.transforms = transforms
        self.dataset = path.split(os.sep)[-2]

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, str]]]:
        """
        Returns image and target (boxes, labels, img_number) from dataset.

        Args:
            index: index of datapoint

        Returns:
            image, target

        """
        # load image and targets depending on objective
        if self.objective == "fullimage":
            imgnum = self.data[index].split(os.sep)[-1]
            try:
                img = read_image(f"{self.data[index]}/{imgnum}.jpg") / 255
            except RuntimeError:
                print(f"Problem with image {imgnum}")
                exit()
            if self.dataset in ["BonnData", "GloSat"]:
                target = reversetablerelativebboxes_outer(self.data[index])
            else:
                target = torch.load(f"{self.data[index]}/{imgnum}.pt")
            # print(img.dtype)
        else:
            "not yet implemented since there is no need, left in so it can be added in future"
            pass

        if self.transforms:
            img = self.transforms(img)

        # if img.shape[0]!=3:
        #    print(self.data[index])

        return (
            img,
            {
                "boxes": target,
                "labels": torch.ones(len(target), dtype=torch.int64),
                "img_number": imgnum,
                "index": index,
            },
        )

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            length of the dataset

        """
        return len(self.data)

    def getidx(self, imname: str) -> int:
        """ "
        returns index of given image name in dataset
        """
        return self.data.index(list(filter(lambda d: imname in d, self.data))[0])


if __name__ == "__main__":
    import numpy as np
    from torch.nn import ModuleList, Sequential
    from torchvision import transforms

    transform = Sequential(
        transforms.RandomApply(
            ModuleList(
                [transforms.ColorJitter(brightness=(0.5, 1.5), saturation=(0, 2))]
            ),
            p=0,
        ),
        transforms.RandomApply(
            ModuleList([transforms.GaussianBlur(kernel_size=9, sigma=(2, 10))]),
            p=0,
        ),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0),
        transforms.RandomGrayscale(p=1),
    )

    dataset = CustomDataset(
        f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/train",
        "fullimage",
        transforms=None,
    )

    # dataset = CustomDataset(
    #    f"{Path(__file__).parent.absolute()}/../../data/BonnData/train",
    #    "fullimage",
    #    transforms=None,
    # )

    for i in tqdm(range(dataset.__len__())):
        img, target = dataset[i]
        if img.shape[0] != 3:
            print(img.shape)
            print(target["img_number"])

    # shutil.copy(src=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/rawdata/train/images/0SlmUiAXS2SAJ-PLom0zagAAACMAAQED.jpg", dst=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/train/0SlmUiAXS2SAJ-PLom0zagAAACMAAQED/0SlmUiAXS2SAJ-PLom0zagAAACMAAQED.jpg")
    # img, target = dataset[dataset.getidx("mit_google_image_search-10918758-be4b5fa7bf3fea80823dabbe1e17e4136f0da811")]
    # print(target['boxes'])
    # img,target = dataset[dataset.getidx('customs-declaration-12689')]
    # print(target['boxes'])
    # dataset = CustomDataset(
    #   f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/train",
    #    "fullimage",
    #    transforms=None,
    # )

    img, target = dataset[
        dataset.getidx(
            "mit_google_image_search-10918758-cdcd82db9ce0b61da60155c5c822b0be3884a2cf"
        )
    ]

    # img, target = dataset[0]
    # print(target['boxes'])
    # result = draw_bounding_boxes(
    #    image=(img * 255).to(torch.uint8),
    #    boxes=target["boxes"],
    #    colors=["red" for i in range(target["boxes"].shape[0])],
    #    labels=["Ground" for i in range(target["boxes"].shape[0])],
    # )
    # result = Image.fromarray(result.permute(1, 2, 0).numpy())
    # result.save(f"{Path(__file__).parent.absolute()}/../../images/rcnn/testest_4.jpg")
