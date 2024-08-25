"""
Code modified from https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb
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
import argparse
import os
from pathlib import Path

import lightning.pytorch as pl
from lightning.fabric.utilities import move_data_to_device
from lightning.pytorch.utilities.types import EVAL_DATALOADERS

from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import draw_bounding_boxes
from transformers import TableTransformerForObjectDetection
import torch
from lightning.pytorch import loggers, Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

from src.historicdocumentprocessing.tabletransformer_dataset import CustomDataset


class TableTransformer(pl.LightningModule):
     def __init__(self, lr, lr_backbone, weight_decay, testdataset: CustomDataset, traindataset:CustomDataset, valdataset:CustomDataset=None, datasetname:str = "BonnData"):
         super().__init__()
         self.model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition").to(self.device)
         # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
         self.lr = lr
         self.lr_backbone = lr_backbone
         self.weight_decay = weight_decay
         self.train_dataset=traindataset
         self.val_dataset=valdataset
         self.test_dataset=testdataset
         self.writer = None
         if datasetname == "Tablesinthewild" and valdataset:
            self.example_image, self.example_target = valdataset.getimgtarget(
                 valdataset.getidx("mit_google_image_search-10918758-be4b5fa7bf3fea80823dabbe1e17e4136f0da811"))
            self.train_example_image, self.train_example_target = traindataset.getimgtarget(
                 traindataset.getidx("mit_google_image_search-10918758-cdcd82db9ce0b61da60155c5c822b0be3884a2cf"))
         elif datasetname == "Tablesinthewild" and not valdataset:
             self.example_image, self.example_target = validdataset.getimgtarget(0)
             self.train_example_image, self.train_example_target = traindataset.getimgtarget(
                 traindataset.getidx("mit_google_image_search-10918758-cdcd82db9ce0b61da60155c5c822b0be3884a2cf"))
         else:
             self.example_image, self.example_target = validdataset.getimgtarget(0)
             self.train_example_image, self.train_example_target = traindataset.getimgtarget(0)
         self.example_image = (self.example_image)
         self.train_example_image = (self.train_example_image)

     def forward(self, pixel_values, pixel_mask):
       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

       return outputs

     def common_step(self, batch, batch_idx):
       pixel_values = batch["pixel_values"].to(self.device)
       pixel_mask = batch["pixel_mask"].to(self.device)
       labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

       loss = outputs.loss
       loss_dict = outputs.loss_dict

       return loss, loss_dict

     def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss, on_epoch=True)
        for k,v in loss_dict.items():
          self.log("train_" + k, v.item(), on_epoch=True)

        return loss

     def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True)
        for k,v in loss_dict.items():
          self.log("validation_" + k, v.item(), on_epoch=True)

     def on_validation_epoch_end(self) -> None:
        for logger in self.trainer.loggers:
             if isinstance(logger, loggers.TensorBoardLogger):
                 self.writer = logger.experiment
        #trainexampleimg
        self.model.eval()
        encoding = move_data_to_device(self.train_dataset.ImageProcessor(self.train_example_image, return_tensors="pt"), self.device)
        with torch.no_grad():
            out = self.model(**encoding)
        width, height = self.train_example_image.size
        pred = self.train_dataset.ImageProcessor.post_process_object_detection(out,target_sizes=[(height, width)])[0]
        self.model.train()
        boxes = {
            "ground truth": self.train_example_target,
            "prediction": pred["boxes"].detach().cpu(),
        }
        colors = ["green" for i in range(boxes["prediction"].shape[0])] + [
            "red" for i in range(boxes["ground truth"].shape[0])
        ]
        labels = ["Pred" for i in range(boxes["prediction"].shape[0])] + [
            "Ground" for i in range(boxes["ground truth"].shape[0])
        ]
        result = draw_bounding_boxes(
            image=pil_to_tensor(self.train_example_image).to(torch.uint8),
            boxes=torch.vstack((boxes["prediction"], boxes["ground truth"])),
            colors=colors,
            labels=labels,
        )
        self.writer.add_image(
            "Train/example", result[:, ::2, ::2], global_step=self.global_step
        )

        ##validexampleimg
        self.model.eval()
        encoding = move_data_to_device(self.val_dataset.ImageProcessor(self.example_image, return_tensors="pt"), self.device)

        with torch.no_grad():
            out = self.model(**encoding)
        width, height = self.example_image.size
        pred = self.val_dataset.ImageProcessor.post_process_object_detection(out, target_sizes=[(height, width)])[0]
        self.model.train()
        boxes = {
            "ground truth": self.example_target,
            "prediction": pred["boxes"].detach().cpu(),
        }
        colors = ["green" for i in range(boxes["prediction"].shape[0])] + [
            "red" for i in range(boxes["ground truth"].shape[0])
        ]
        labels = ["Pred" for i in range(boxes["prediction"].shape[0])] + [
            "Ground" for i in range(boxes["ground truth"].shape[0])
        ]
        result = draw_bounding_boxes(
            image=pil_to_tensor(self.example_image).to(torch.uint8),
            boxes=torch.vstack((boxes["prediction"], boxes["ground truth"])),
            colors=colors,
            labels=labels,
        )
        self.writer.add_image(
            "Valid/example", result[:, ::2, ::2], global_step=self.global_step
        )

     def configure_optimizers(self):
        param_dicts = [
              {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                  "lr": self.lr_backbone,
              },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                  weight_decay=self.weight_decay)

        return optimizer

     def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, collate_fn=self.val_dataset.collate_fn)

     def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate_fn,num_workers=15)

     def test_dataloader(self) -> EVAL_DATALOADERS:
         return DataLoader(self.test_dataset, batch_size=1, collate_fn=self.val_dataset.collate_fn)


def get_args() -> argparse.Namespace:
    """Defines arguments."""
    parser = argparse.ArgumentParser(description="training")

    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default="model",
        help="Name of the model, for saving and logging",
    )

    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=250,
        help="Number of epochs",
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="Tablesinthewild",
        help="which dataset should be used for training",
    )

    parser.add_argument(
        "--objective",
        "-o",
        type=str,
        default="fullimage",
        help="objective of the model ('fullimage')",
    )

    parser.add_argument(
        "--load",
        "-l",
        type=str,
        default=None,
        help="name of a model to load",
    )

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)


    parser.add_argument(
        "--startepoch", "-se", type=int, default=1, help="number of starting epoch"
    )
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=1)

    # parser.add_argument('--augmentations', "-a", action=argparse.BooleanOptionalAction)
    parser.add_argument("--augmentations", action="store_true")
    parser.add_argument(
        "--no-augmentations", dest="augmentations", action="store_false"
    )
    parser.set_defaults(augmentations=False)

    parser.add_argument("--valid", action="store_true")
    parser.add_argument("--no-valid", dest="valid", action="store_false")
    parser.set_defaults(valid=True)

    parser.add_argument("--early_stopping", "-es", action="store_true")
    parser.add_argument("--no-early_stopping","-no_es", dest="early_stopping", action="store_false")
    parser.set_defaults(early_stopping=False)

    return parser.parse_args()

if __name__=='__main__':
    args = get_args()
    name = (
        f"tabletransformer_{args.name}_{args.dataset}_{args.objective}"
        f"{'_aug' if args.augmentations else ''}_e{args.epochs}"
        f"{f'_init_{args.load_}' if args.load else ''}"
    #    f"{'_random_init' if args.randominit else ''}"
        f"{'_valid' if args.valid else '_no_valid'}"
    )
    train_log_dir = (
        f"{Path(__file__).parent.absolute()}/../../logs/runs/"
    )
    traindataset = CustomDataset(
        f"{Path(__file__).parent.absolute()}/../../data/{args.dataset}/train",
        args.objective
    )

    print("start training:")
    print(f"\tname: {name}")
    print(f"\tobjective: {args.objective}")
    print(f"\tdataset: {args.dataset}")
    print(f"\tepochs: {args.epochs}")
    print(f"\tload: {args.load}\n")
    #print(f"\trandom initialization: {args.randominit}\n")
    print(f"Use Valid Set: {args.valid}")

    if args.valid:
        validdataset = CustomDataset(
            f"{Path(__file__).parent.absolute()}/../../data/{args.dataset}/valid", args.objective
        )
    #    valid_dataloader = DataLoader(dataset=validdataset)
    else:
        validdataset = None
    #    valid_dataloader = None
    #train_dataloader = DataLoader(dataset=traindataset)
    testdataset = CustomDataset(f"{Path(__file__).parent.absolute()}/../../data/{args.dataset}/test", args.objective)

    print(f"{len(traindataset)=}")
    print(f"{len(validdataset)=}")

    checkpoint_callback = ModelCheckpoint(dirpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/tabletransformer/", filename=f"{f'{name}_es' if args.early_stopping else f'{name}_end'}")
    logger = loggers.TensorBoardLogger(save_dir=train_log_dir, name=name)
    model = TableTransformer(lr=args.lr, lr_backbone=args.lr_backbone, weight_decay=args.weight_decay, traindataset=traindataset, valdataset=validdataset, testdataset=testdataset, datasetname=args.dataset)
    if args.valid and not args.early_stopping:
        trainer = Trainer(logger=logger,max_epochs=args.epochs, devices=args.gpus, accelerator="cuda", num_nodes=args.num_nodes, callbacks=[checkpoint_callback])
        trainer.fit(model)
    elif args.valid and args.early_stopping:
        trainer = Trainer(logger=logger,max_epochs=args.epochs, devices=args.gpus, num_nodes=args.num_nodes,  callbacks=[EarlyStopping(monitor="training_loss", mode="min"), checkpoint_callback])
        trainer.fit(model)
    else:
        #no validation
        trainer = Trainer(logger=logger,max_epochs=args.epochs, devices=args.gpus, accelerator="cuda", num_nodes=args.num_nodes, limit_val_batches=0, num_sanity_val_steps=0, callbacks=[checkpoint_callback])
        trainer.fit(model)
    #trainer.fit(model, train_dataloader)

