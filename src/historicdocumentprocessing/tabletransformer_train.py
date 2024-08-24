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
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import TableTransformerForObjectDetection
import torch
from pytorch_lightning import loggers, Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

from src.historicdocumentprocessing.tabletransformer_dataset import CustomDataset


class TableTransformer(pl.LightningModule):
     def __init__(self, lr, lr_backbone, weight_decay, train_dataloader, val_dataloader):
         super().__init__()
         self.model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition").to(self.device)
         # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
         self.lr = lr
         self.lr_backbone = lr_backbone
         self.weight_decay = weight_decay
         self.train_dataloader=train_dataloader
         self.val_dataloader=val_dataloader
         self.writer = self.logger.experiment

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
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
          self.log("train_" + k, v.item())

        return loss

     def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
          self.log("validation_" + k, v.item())

        return loss

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
        return self.train_dataloader

     def val_dataloader(self):
        return self.val_dataloader

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
        f"{args.name}_{args.dataset}_{args.objective}"
        f"{'_aug' if args.augmentations else ''}_e{args.epochs}"
        f"{f'_init_{args.load_}' if args.load else ''}"
    #    f"{'_random_init' if args.randominit else ''}"
        f"{'valid' if args.valid else 'no_valid'}"
    )
    train_log_dir = (
        f"{Path(__file__).parent.absolute()}/../../logs/runs/{name}"
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
    print(f"\trandom initialization: {args.randominit}\n")
    print(f"Use Valid Set: {args.valid}")

    if args.valid:
        validdataset = CustomDataset(
            f"{Path(__file__).parent.absolute()}/../../data/{args.dataset}/valid", args.objective
        )
        valid_dataloader = DataLoader(dataset=validdataset)
    else:
        valid_dataloader = None
    train_dataloader = DataLoader(dataset=traindataset)

    checkpoint_callback = ModelCheckpoint(dirpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/tabletransformer/", filename=f"{f'{name}_es' if args.early_stopping else f'{name}_end'}")
    model = TableTransformer(lr=args.lr, lr_backbone=args.lr_backbone, weight_decay=args.weight_decay, train_dataloader=train_dataloader, val_dataloader=valid_dataloader)
    logger = loggers.TensorBoardLogger(train_log_dir)
    if args.valid and not args.early_stopping:
        trainer = Trainer(logger=logger,max_epochs=args.epochs, gpus=args.gpus, num_nodes=args.num_nodes, callbacks=[checkpoint_callback])
        trainer.fit(model, train_dataloader, valid_dataloader)
    elif args.valid and args.early_stopping:
        trainer = Trainer(logger=logger,max_epochs=args.epochs, gpus=args.gpus, num_nodes=args.num_nodes,  callbacks=[EarlyStopping(monitor="training_loss", mode="min"), checkpoint_callback])
        trainer.fit(model, train_dataloader, valid_dataloader)
    else:
        #no validation
        trainer = Trainer(logger=logger,max_epochs=args.epochs, gpus=args.gpus, num_nodes=args.num_nodes, limit_val_batches=0, num_sanity_val_steps=0, callbacks=[checkpoint_callback])
        trainer.fit(model, train_dataloader, val_dataloaders=None)
    #trainer.fit(model, train_dataloader)

