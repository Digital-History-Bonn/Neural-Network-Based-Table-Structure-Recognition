"""Dataset Class for training. Code from
https://github.com/Digital-History-Bonn/HistorischeTabellenSemanticExtraction/blob/main/src/TableExtraction
/customdataset.py"""

import glob
import os
from pathlib import Path
from typing import Tuple, Union, Dict, Optional

import torch
from torch.nn import Module
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):   # type: ignore
    """Dataset Class for training."""

    def __init__(self, path: str, objective: str, transforms: Optional[Module] = None) -> None:
        """
        Dataset Class for training.

        Args:
            path: path to folder with images
            objective: detection object ('table', 'cell', 'row' or 'col')
            transforms: torchvision transforms for on-the-fly augmentations
        """
        super().__init__()
        if objective == "table":
            self.data = sorted(list(glob.glob(f"{path}/*")), key=lambda x: int(x.split(os.sep)[-1]))
        else:
            self.data = list(glob.glob(f"{path}/*/*_table_*.pt"))
        self.objective = objective
        self.transforms = transforms

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, str]]]:
        """
        Returns image and target (boxes, labels, img_number) from dataset.

        Args:
            index: index of datapoint

        Returns:
            image, target

        """
        # load image and targets depending on objective
        if self.objective == "table":
            imgnum = self.data[index].split(os.sep)[-1]
            img = torch.load(f"{self.data[index]}/{imgnum}.pt") / 256
            target = torch.load(f"{self.data[index]}/{imgnum}_tables.pt")
        else:
            imgnum = self.data[index].split(os.sep)[-2]
            tablenum = self.data[index].split(os.sep)[-1].split("_")[-1][-4]
            img = torch.load(self.data[index]) / 256
            target = torch.load(
                f"{'/'.join(self.data[index].split(os.sep)[:-1])}/{imgnum}"
                f"_{self.objective}_{tablenum}.pt"
            )

        if self.transforms:
            img = self.transforms(img)

        return (
            img,
            {
                "boxes": target,
                "labels": torch.ones(len(target), dtype=torch.int64),
                "img_number": imgnum,
            },
        )

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            length of the dataset

        """
        return len(self.data)


if __name__ == "__main__":
    import numpy as np
    from torch.nn import Sequential, ModuleList
    from torchvision import transforms

    transform = Sequential(
        transforms.RandomApply(
            ModuleList(
                [transforms.ColorJitter(brightness=(0.5, 1.5), saturation=(0, 2))]
            ),
            p=0,
        ),
        transforms.RandomApply(
            ModuleList(
                [transforms.GaussianBlur(kernel_size=9, sigma=(2, 10))]
            ),
            p=0,
        ),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0),
        transforms.RandomGrayscale(p=1),
    )

    dataset = CustomDataset(
        f"{Path(__file__).parent.absolute()}/../data/Tables/preprocessed/",
        "tables",
        transforms=transform,
    )

    img, target = dataset[3]

    result = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

    result.save(
        f"{Path(__file__).parent.absolute()}/../data/assets/"
        f"Originals_SampleImage.png"
    )