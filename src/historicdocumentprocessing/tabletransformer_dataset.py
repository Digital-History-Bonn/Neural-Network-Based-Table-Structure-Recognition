"""Dataset Class for training. Code  modified from
https://github.com/Digital-History-Bonn/HistorischeTabellenSemanticExtraction/blob/main/src/TableExtraction/customdataset.py

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
from typing import Dict, Optional, Tuple, Union, List
from unicodedata import category

import torch
from PIL import Image
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import torchvision.ops as ops
from transformers import DetrImageProcessor, BatchFeature
from tqdm import tqdm


from src.historicdocumentprocessing.kosmos_eval import reversetablerelativebboxes_outer


class CustomDataset(Dataset):  # type: ignore
    """Dataset Class for training."""

    def __init__(
        self, path: str, objective: str = "fullimage", transforms: Optional[Module] = None) -> None:
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
        self.ImageProcessor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.objective = objective
        self.transforms = transforms
        self.dataset = path.split(os.sep)[-2]
        #maximsize=[0,0]
        #for d in self.data:
        #    imgnum = d.split(os.sep)[-1]
        #    imsize = Image.open(f"{d}/{imgnum}.jpg").size
        #    maximsize = [max(maximsize[0], imsize[0]), max(maximsize[1], imsize[1])]
        #self.padsize = {"height":maximsize[0], "width":maximsize[1]}
        #self.padsize = {"height": 800, "width": 1333}
        #print(self.padsize)

    def __getitem__(
        self, index: int
    ) -> BatchFeature:
        """
        Returns image and target (boxes, labels, img_number) from dataset.

        Args:
            index: index of datapoint

        Returns:
            image, target

        """
        # load image and targets depending on objective
        imgnum = self.data[index].split(os.sep)[-1]
        if index==0:
            print(imgnum)
        if self.objective == "fullimage":
            img = Image.open(f"{self.data[index]}/{imgnum}.jpg").convert("RGB")
            if self.dataset in ["BonnData", "GloSat"]:
                target = reversetablerelativebboxes_outer(self.data[index])
            else:
                target = torch.load(f"{self.data[index]}/{imgnum}.pt")
            # print(img.dtype)
        else:
            "not yet implemented since there is no need, left in so it can be added in future"
            target=None
            img=None
            pass

        #get bbox area
        area = ops.box_area(target)
        #convert bboxes to coco format
        target = ops.box_convert(target, in_fmt='xyxy', out_fmt='xywh')
        if self.transforms:
            img = self.transforms(img)
        assert area.shape[0]==target.shape[0]
        annotations = {'image_id':index, 'annotations': [{'image_id':index, 'category_id':0, "iscrowd":0, "area": area[i], "bbox": target[i].tolist()} for i in range(target.shape[0])]}
        #if self.dataset=="BonnData" and include_textregions
        encoding = self.ImageProcessor(images=img, annotations=annotations, no_pad=True)
        # if img.shape[0]!=3:
        #    print(self.data[index])
        #print(encoding.keys())
        #print(encoding["pixel_values"])
        #print(encoding["labels"])

        return encoding

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

    def collate_fn(self, batch):
        pixelvalues = [b["pixel_values"][0] for b in batch]
        encoding = self.ImageProcessor.pad(pixelvalues, return_tensors="pt")
        return {"pixel_values": encoding["pixel_values"], "pixel_mask": encoding["pixel_mask"], "labels": [b["labels"][0] for b in batch]}



if __name__ == "__main__":

    dataset = CustomDataset(
        f"{Path(__file__).parent.absolute()}/../../data/BonnData/train",
        "fullimage",
        transforms=None,
    )
    encoding = dataset[0]
    print(encoding.keys())
    #print([encoding[key] for key in encoding.keys()])
    print(encoding["pixel_values"][0].shape)
    dataloader = DataLoader(dataset=dataset, batch_size=4, collate_fn=dataset.collate_fn)
    batch = next(iter(dataloader))
    print(batch.keys())
    print(batch)
