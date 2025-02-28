"""Training Code for Tabletransformer.

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
import statistics
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.fabric.utilities import move_data_to_device
from lightning.pytorch import Trainer, loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import draw_bounding_boxes
from transformers import AutoModelForObjectDetection

from src.historicdocumentprocessing.tabletransformer_dataset import CustomDataset


class TableTransformer(pl.LightningModule):
    """Class to train TableTransformer models."""
    def __init__(
        self,
        lr,
        lr_backbone,
        weight_decay,
        testdataset: CustomDataset,
        traindataset: CustomDataset,
        valdataset: CustomDataset = None,
        datasetname: str = "BonnData",
        savepath: str = None,
        loadmodelcheckpoint: str = None,
    ):
        """Class to train TableTransformer models.

        Args:
            lr: Learning Rate
            lr_backbone: Learning Rate for Backbone Network
            weight_decay: Weight Decay
            testdataset: dataset to test on
            traindataset: dataset to train on
            valdataset: dataset to validate
            datasetname: name of dataset (BonnData, Tablesinthewild, GloSAT)
            savepath: path to model save folder
            loadmodelcheckpoint: name of model to load
        """
        super().__init__()
        # self.model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition-v1.1-all").to(self.device)
        self.model = AutoModelForObjectDetection.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        ).to(self.device)
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        # print(model.config.id2label)
        if loadmodelcheckpoint:
            self.model.load_state_dict(
                torch.load(
                    f"{Path(__file__).parent.absolute()}/../../checkpoints/tabletransformer/{loadmodelcheckpoint}.pt"
                )
            )
            self.model.train()
            print("loaded: ", loadmodelcheckpoint)
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.train_dataset = traindataset
        self.val_dataset = valdataset
        self.test_dataset = testdataset
        self.writer = None
        if datasetname == "Tablesinthewild" and valdataset:
            self.example_image, self.example_target, self.example_lable = (
                valdataset.getimgtarget(
                    valdataset.getidx(
                        "mit_google_image_search-10918758-bf3a1b339a99b2d38e05fb70dfb8afa3e4db9e3c"
                    )
                )
            )
            (
                self.train_example_image,
                self.train_example_target,
                self.train_example_lable,
            ) = traindataset.getimgtarget(
                traindataset.getidx(
                    "mit_google_image_search-10918758-cdcd82db9ce0b61da60155c5c822b0be3884a2cf"
                )
            )
        elif datasetname == "Tablesinthewild" and not valdataset:
            self.example_image, self.example_target, self.example_lable = (
                valdataset.getimgtarget(0)
            )
            (
                self.train_example_image,
                self.train_example_target,
                self.train_example_lable,
            ) = traindataset.getimgtarget(
                traindataset.getidx(
                    "mit_google_image_search-10918758-cdcd82db9ce0b61da60155c5c822b0be3884a2cf"
                )
            )
        else:
            self.example_image, self.example_target, self.example_lable = (
                valdataset.getimgtarget(0)
            )
            (
                self.train_example_image,
                self.train_example_target,
                self.train_example_lable,
            ) = traindataset.getimgtarget(0)
        self.example_image = self.example_image
        self.train_example_image = self.train_example_image
        self.savepath = savepath
        self.val_losses = []
        self.mean_val_loss = None

    def forward(self, pixel_values, pixel_mask):
        """Forward method.

        Args:
            pixel_values: pixel values
            pixel_mask: pixel mask

        Returns:
            model outputs

        """
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        return outputs

    def common_step(self, batch, batch_idx):
        """Common parts of train and valid steps.

        Args:
            batch: output of the dataloader
            batch_idx: index of the batch

        Returns:
            training loss and loss_dict

        """
        pixel_values = batch["pixel_values"].to(self.device)
        pixel_mask = batch["pixel_mask"].to(self.device)
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(
            pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels
        )

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        """Training step, calculate train loss.

        Args:
            batch: output of the dataloader
            batch_idx: index of the batch

        Returns:
            training loss

        """
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss, on_epoch=True)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item(), on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Valid step, calculate valid loss.

        Args:
            batch: output of the dataloader
            batch_idx: index of the batch

        Returns:
            valid loss
        """
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item(), on_epoch=True)
        self.val_losses.append(loss.detach().item())
        return loss

    def on_validation_epoch_end(self) -> None:
        """Record validation metrics and sample images at end of validation epoch, also save model state dict and do early stopping."""
        mean_val_loss = statistics.mean(self.val_losses)
        if not self.mean_val_loss or self.mean_val_loss > mean_val_loss:
            self.mean_val_loss = mean_val_loss
            torch.save(self.model.state_dict(), self.savepath + "_es.pt")
        self.val_losses.clear()
        for logger in self.trainer.loggers:
            if isinstance(logger, loggers.TensorBoardLogger):
                self.writer = logger.experiment
        # trainexampleimg
        self.model.eval()
        torch.save(self.model.state_dict(), self.savepath + "_end.pt")
        encoding = move_data_to_device(
            self.train_dataset.ImageProcessor(
                self.train_example_image, return_tensors="pt"
            ),
            self.device,
        )
        with torch.no_grad():
            out = self.model(**encoding)
        width, height = self.train_example_image.size
        pred = self.train_dataset.ImageProcessor.post_process_object_detection(
            out, target_sizes=[(height, width)]
        )[0]
        self.model.train()
        boxes = {
            "ground truth": self.train_example_target,
            "prediction": pred["boxes"].detach().cpu(),
            "labels": pred["labels"].detach().cpu(),
        }
        colors = ["green" for i in range(boxes["prediction"].shape[0])] + [
            "red" for i in range(boxes["ground truth"].shape[0])
        ]
        # labels = ["Pred" for i in range(boxes["prediction"].shape[0])] + [
        #    "Ground" for i in range(boxes["ground truth"].shape[0])
        # ]
        labels = [
            str(self.model.config.id2label[label]) for label in boxes["labels"].tolist()
        ] + ["Ground" for i in range(boxes["ground truth"].shape[0])]

        result = draw_bounding_boxes(
            image=pil_to_tensor(self.train_example_image).to(torch.uint8),
            boxes=torch.vstack((boxes["prediction"], boxes["ground truth"])),
            colors=colors,
            labels=labels,
        )
        self.writer.add_image(
            "Train/example", result[:, ::2, ::2], global_step=self.global_step
        )

        # validexampleimg
        self.model.eval()
        encoding = move_data_to_device(
            self.val_dataset.ImageProcessor(self.example_image, return_tensors="pt"),
            self.device,
        )

        with torch.no_grad():
            out = self.model(**encoding)
        width, height = self.example_image.size
        pred = self.val_dataset.ImageProcessor.post_process_object_detection(
            out, target_sizes=[(height, width)]
        )[0]
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
        """Method to configure the optimizer.

        Returns:
            optimizer
        """
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(
            param_dicts, lr=self.lr, weight_decay=self.weight_decay
        )

        return optimizer

    def train_dataloader(self):
        """Get train dataloader.

        Returns:
            train dataloader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=1,
            collate_fn=self.train_dataset.collate_fn,
            num_workers=2,
        )

    def val_dataloader(self):
        """Get validation dataloader.

        Returns:
            validation dataloader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            collate_fn=self.val_dataset.collate_fn,
            num_workers=2,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Get test dataloader.

        Returns:
            test dataloader
        """
        return DataLoader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate_fn
        )


def get_args() -> argparse.Namespace:
    """Defines arguments."""   # noqa: DAR201
    parser = argparse.ArgumentParser(description="tabletransformer_train")

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
    parser.add_argument(
        "--no-early_stopping", "-no_es", dest="early_stopping", action="store_false"
    )
    parser.set_defaults(early_stopping=False)

    parser.add_argument("--identicalname", action="store_true")
    parser.set_defaults(identicalname=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    name = (
        f"tabletransformer_v0_new_{args.name}_{args.dataset}_{args.objective}"
        f"{'_aug' if args.augmentations else ''}_e{args.epochs}"
        f"{f'_init_{args.load}' if args.load else ''}"
        #    f"{'_random_init' if args.randominit else ''}"
        f"{'_valid' if args.valid else '_no_valid'}"
    )
    # if args.load:
    #     modelname = args.load.split("/")[-1]
    #     name = f"{name}_loaded_{modelname}"
    train_log_dir = f"{Path(__file__).parent.absolute()}/../../logs/runs/"
    traindataset = CustomDataset(
        f"{Path(__file__).parent.absolute()}/../../data/{args.dataset}/train",
        args.objective,
    )

    if args.identicalname:
        name = args.name
    print("start training:")
    print(f"\tname: {name}")
    print(f"\tobjective: {args.objective}")
    print(f"\tdataset: {args.dataset}")
    print(f"\tepochs: {args.epochs}")
    print(f"\tload: {args.load}\n")
    # print(f"\trandom initialization: {args.randominit}\n")
    print(f"Use Valid Set: {args.valid}")
    print("Cuda Available:", torch.cuda.is_available())
    print("torch lightning early_stopping:", args.early_stopping)

    if args.valid:
        validdataset = CustomDataset(
            f"{Path(__file__).parent.absolute()}/../../data/{args.dataset}/valid",
            args.objective,
        )
    #    valid_dataloader = DataLoader(dataset=validdataset)
    else:
        validdataset = None
    #    valid_dataloader = None
    # train_dataloader = DataLoader(dataset=traindataset)
    testdataset = CustomDataset(
        f"{Path(__file__).parent.absolute()}/../../data/{args.dataset}/test",
        args.objective,
    )

    print(f"{len(traindataset)=}")
    print(f"{len(validdataset)=}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/tabletransformer/",
        filename=f"{f'{name}_es' if args.early_stopping else f'{name}_end'}",
    )
    logger = loggers.TensorBoardLogger(
        save_dir=train_log_dir, name=name, version="severalcalls"
    )  # version needs to be constant for same log in case of resuming training
    model = TableTransformer(
        lr=args.lr,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
        traindataset=traindataset,
        valdataset=validdataset,
        testdataset=testdataset,
        datasetname=args.dataset,
        savepath=f"{Path(__file__).parent.absolute()}/../../checkpoints/tabletransformer/{name}",
        loadmodelcheckpoint=args.load,
    )
    if args.identicalname and args.load and args.valid and not args.early_stopping:
        trainer = Trainer(
            logger=logger,
            max_epochs=args.epochs,
            devices=args.gpus,
            accelerator="cuda",
            num_nodes=args.num_nodes,
            callbacks=[checkpoint_callback],
        )
        checkpoint = torch.load(
            f"{Path(__file__).parent.absolute()}/../../checkpoints/tabletransformer/{args.load}.ckpt"
        )
        print("global_step", checkpoint["global_step"])
        trainer.fit(
            model,
            ckpt_path=f"{Path(__file__).parent.absolute()}/../../checkpoints/tabletransformer/{args.load}.ckpt",
        )
    elif args.valid and not args.early_stopping:
        trainer = Trainer(
            logger=logger,
            max_epochs=args.epochs,
            devices=args.gpus,
            accelerator="cuda",
            num_nodes=args.num_nodes,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(model)
    elif args.valid and args.early_stopping:
        trainer = Trainer(
            logger=logger,
            max_epochs=args.epochs,
            devices=args.gpus,
            num_nodes=args.num_nodes,
            callbacks=[
                EarlyStopping(monitor="training_loss", mode="min"),
                checkpoint_callback,
            ],
        )
        trainer.fit(model)
    else:
        # no validation
        trainer = Trainer(
            logger=logger,
            max_epochs=args.epochs,
            devices=args.gpus,
            accelerator="cuda",
            num_nodes=args.num_nodes,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(model)
    # trainer.fit(model, train_dataloader)
