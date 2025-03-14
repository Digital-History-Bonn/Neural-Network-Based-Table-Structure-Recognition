"""Dataset Class for training. Code  modified from https://github.com/Digital-History-Bonn/HistorischeTabellenSemanticExtraction/blob/main/src/TableExtraction/customdataset.py .

made using https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DETR as a guideline

License:
MIT License

Copyright (c) 2021 NielsRogge

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import glob
import os
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import torch
import torchvision.ops as ops
from lightning.pytorch.core.hooks import move_data_to_device
from PIL import Image
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import draw_bounding_boxes
from transformers import AutoImageProcessor, BatchFeature

from src.historicdocumentprocessing.util.tablesutil import reversetablerelativebboxes_inner, \
    reversetablerelativebboxes_outer


def reversetablerelativebboxes_outer_rowcoll(
    fpath: str, category: Literal["row", "col"]
) -> torch.Tensor:
    """Get BBoxes relative to image from row/column BBoxes relative to table.

    Args:
        fpath: path to table folder
        category: row or column

    Returns:
        BBox coordinates relative to image

    """
    tablebboxes = torch.load(glob.glob(f"{fpath}/*tables.pt")[0])
    newcoords = torch.zeros((0, 4))
    for table in glob.glob(f"{fpath}/*_{category}_*.pt"):
        n = int(table.split(".")[-2].split("_")[-1])
        newcell = reversetablerelativebboxes_inner(tablebbox=tablebboxes[n], cellbboxes=torch.load(table))
        newcoords = torch.vstack((newcoords, newcell))
    return newcoords


class CustomDataset(Dataset):  # type: ignore
    """Dataset Class for training."""

    def __init__(
        self,
        path: str,
        objective: str = "fullimage",
        transforms: Optional[Module] = None,
    ) -> None:
        """Dataset Class for training.

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
        # self.ImageProcessor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        # see https://github.com/microsoft/table-transformer/blob/16d124f616109746b7785f03085100f1f6247575/src/inference.py#L45 for table transformer size and img mean,std values
        # self.ImageProcessor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition-v1.1-all", size={"shortest_edge":1000, "longest_edge":1000})
        # self.ImageProcessor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition-v1.1-all", size={"shortest_edge":800, "longest_edge":1000})
        self.ImageProcessor = AutoImageProcessor.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        )
        # print(self.ImageProcessor.size['longest_edge'],self.ImageProcessor.size['shortest_edge'])
        # print(self.ImageProcessor.image_mean, self.ImageProcessor.image_std)
        # assert self.ImageProcessor.size['longest_edge']==self.ImageProcessor.size['shortest_edge']==1000
        # assert self.ImageProcessor.size['longest_edge'] == 1000 and self.ImageProcessor.size['shortest_edge'] == 800
        assert self.ImageProcessor.image_mean == [0.485, 0.456, 0.406]
        assert self.ImageProcessor.image_std == [0.229, 0.224, 0.225]
        self.objective = objective
        self.transforms = transforms
        self.dataset = path.split(os.sep)[-2]
        if self.dataset not in ["BonnData", "GloSat", "Tablesinthewild"]:
            self.dataset = path.split(os.sep)[-3]
        # maximsize=[0,0]
        # for d in self.data:
        #    imgnum = d.split(os.sep)[-1]
        #    imsize = Image.open(f"{d}/{imgnum}.jpg").size
        #    maximsize = [max(maximsize[0], imsize[0]), max(maximsize[1], imsize[1])]
        # self.padsize = {"height":maximsize[0], "width":maximsize[1]}
        # self.padsize = {"height": 800, "width": 1333}
        # print(self.padsize)

    def __getitem__(self, index: int) -> BatchFeature:
        """Returns image encoding from dataset.

        Args:
            index: index of datapoint

        Returns:
            encoding

        """
        # load image and targets depending on objective
        imgnum = self.data[index].split(os.sep)[-1]
        if index == 0:
            print(imgnum)
        annotations = {"image_id": index, "annotations": []}
        if self.objective == "fullimage":
            img = Image.open(f"{self.data[index]}/{imgnum}.jpg").convert("RGB")
            if self.dataset in ["BonnData", "GloSat", "Tablesinthewild"]:
                for id, cat in zip([2, 1], ["row", "col"]):
                    target = reversetablerelativebboxes_outer_rowcoll(
                        self.data[index], category=cat  # type: ignore
                    )
                    # get bbox area
                    area = ops.box_area(target)
                    # convert bboxes to coco format
                    target = ops.box_convert(target, in_fmt="xyxy", out_fmt="xywh")
                    assert area.shape[0] == target.shape[0]
                    annotations["annotations"] += [  # type: ignore
                        {
                            "image_id": index,
                            "category_id": id,
                            "iscrowd": 0,
                            "area": area[i],
                            "bbox": target[i].tolist(),
                        }
                        for i in range(target.shape[0])
                    ]
                tables = torch.load(glob.glob(f"{self.data[index]}/*tables.pt")[0])
                area = ops.box_area(tables)
                # convert bboxes to coco format
                target = ops.box_convert(tables, in_fmt="xyxy", out_fmt="xywh")
                assert area.shape[0] == target.shape[0]
                annotations["annotations"] += [  # type: ignore
                    {
                        "image_id": index,
                        "category_id": 0,
                        "iscrowd": 0,
                        "area": area[i],
                        "bbox": target[i].tolist(),
                    }
                    for i in range(tables.shape[0])
                ]
            else:
                target = torch.load(f"{self.data[index]}/{imgnum}.pt")
            # print
        else:
            "not yet implemented since there is no need, left in so it can be added in future"
            target = None
            img = None
            pass

        # get bbox area
        if self.transforms:
            img = self.transforms(img)
        encoding = self.ImageProcessor(
            images=img, annotations=annotations, return_tensors="pt"
        )

        return encoding

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            length of the dataset

        """
        return len(self.data)

    def getidx(self, imname: str) -> int:
        """Returns index of given image name in dataset.

        Args:
            imname: name of image

        Returns:
            index for given image

        """
        return self.data.index(list(filter(lambda d: imname in d, self.data))[0])

    def collate_fn(self, batch):
        """Function needed for batching.

        Args:
            batch: input batch

        Returns:
            padded batch

        """
        pixelvalues = [b["pixel_values"][0] for b in batch]
        encoding = self.ImageProcessor.pad(pixelvalues, return_tensors="pt")
        return {
            "pixel_values": encoding["pixel_values"],
            "pixel_mask": encoding["pixel_mask"],
            "labels": [b["labels"][0] for b in batch],
        }

    def getcells(self, index, addtables: bool = True):
        """Get cell targets at index, also returns table targets if addtables is True.

        Args:
            index: index
            addtables: wether to include table targets

        Returns:
            targets, labels

        Raises:
            Exception: unexpected dataset exception

        """
        imgnum = self.data[index].split(os.sep)[-1]
        if self.objective == "fullimage":
            # img = Image.open(f"{self.data[index]}/{imgnum}.jpg").convert("RGB")
            bboxes = []
            labels = []
            if self.dataset in ["BonnData", "GloSat", "Tablesinthewild"]:
                target = reversetablerelativebboxes_outer(self.data[index])
                bboxes.append(target)
                labels += ["cell" for i in range(target.shape[0])]
                if addtables:
                    tables = torch.load(glob.glob(f"{self.data[index]}/*tables.pt")[0])
                    labels += ["tables" for i in range(tables.shape[0])]
                    bboxes.append(tables)
                target = torch.vstack(bboxes)
            else:
                target = torch.load(f"{self.data[index]}/{imgnum}.pt")
                raise Exception("unexpected Dataset")
            # print(img.dtype)
        else:
            "not yet implemented since there is no need, left in so it can be added in future"
            target = None
            # img = None
            labels = None
            pass
        return target, labels

    def getimgtarget(
        self, index, addtables: bool = True
    ) -> Tuple[Image.Image, torch.Tensor, List]:
        """Get image and row/col target and labels at index also returns table targets if addtables is true.

        Args:
            index: index of image
            addtables: wether to return table targets

        Returns:
            Get image and row/col target and labels at index

        Raises:
            Exception: unexpected dataset exception

        """
        # load image and targets depending on objective
        imgnum = self.data[index].split(os.sep)[-1]
        if self.objective == "fullimage":
            img = Image.open(f"{self.data[index]}/{imgnum}.jpg").convert("RGB")
            bboxes = []
            labels = []
            if self.dataset in ["BonnData", "GloSat", "Tablesinthewild"]:
                for _id, cat in zip([2, 1], ["row", "col"]):
                    target = reversetablerelativebboxes_outer_rowcoll(
                        self.data[index], category=cat  # type: ignore
                    )
                    bboxes.append(target)
                    labels += [f"{cat}" for i in range(target.shape[0])]
                if addtables:
                    tables = torch.load(glob.glob(f"{self.data[index]}/*tables.pt")[0])
                    labels += ["tables" for i in range(tables.shape[0])]
                    bboxes.append(tables)
                target = torch.vstack(bboxes)
            else:
                raise Exception("unexpected Dataset")
        else:
            "not yet implemented since there is no need, left in so it can be added in future"
            target = None
            img = Image.open(f"{self.data[index]}/{imgnum}.jpg").convert("RGB")
            labels = []
            pass
        return img, target, labels

    def getfolder(self, index):
        """Return folder associated with index.

        Args:
            index: index in dataset

        Returns:
            folder associated with index

        """
        return self.data[index]


if __name__ == "__main__":

    # dataset = CustomDataset(
    #    f"{Path(__file__).parent.absolute()}/../../data/BonnData/valid",
    #    "fullimage",
    #    transforms=None,
    # )
    # dataset = CustomDataset(
    #    f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/valid",
    #    "fullimage",
    #    transforms=None,
    # )
    dataset = CustomDataset(
        f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/test",
        "fullimage",
        transforms=None,
    )
    idx = dataset.getidx("000d210d0f9b4101af2006b4aab33c42")
    print(dataset.getimgtarget(idx, addtables=False))

    dataset = CustomDataset(
        f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/Inclined",
        "fullimage",
        transforms=None,
    )
    idx = dataset.getidx("000d210d0f9b4101af2006b4aab33c42")
    print(dataset.getimgtarget(idx, addtables=False))
    encoding = dataset[0]
    print(dataset.getcells(idx, addtables=False))

    dataloader = DataLoader(
        dataset=dataset, batch_size=4, collate_fn=dataset.collate_fn
    )
    batch = next(iter(dataloader))

    device = (
        torch.device(f"cuda:{0}") if torch.cuda.is_available() else torch.device("cpu")
    )
    newbatch = move_data_to_device(batch, device)
    img, target, labels = dataset.getimgtarget(1)
    # result = draw_bounding_boxes(
    #    image=pil_to_tensor(img).to(torch.uint8),
    #    boxes=target,
    #    labels=labels,
    # )
    # img = Image.fromarray(result.permute(1, 2, 0).numpy())
    # img.save(f"{Path(__file__).parent.absolute()}/../../images/test/testtabletransformerdatasettablesinthewild.jpg")
    cells, labels = dataset.getcells(1, addtables=False)
    result = draw_bounding_boxes(
        image=pil_to_tensor(img).to(torch.uint8),
        boxes=cells,
        labels=labels,
    )
    img = Image.fromarray(result.permute(1, 2, 0).numpy())
    # img.save(f"{Path(__file__).parent.absolute()}/../../images/test/testtabletransformerdatasettablesinthewild_cells.jpg")
