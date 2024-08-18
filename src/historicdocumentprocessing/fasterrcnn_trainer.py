"""Module to train models on table datasets. Code  modified from
https://github.com/Digital-History-Bonn/HistorischeTabellenSemanticExtraction/blob/main/src/TableExtraction/trainer
.py"""

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm
from triton.language import dtype

from src.historicdocumentprocessing.fasterrcnn_dataset import CustomDataset

LR = 0.00001


class Trainer:
    """Class to train models."""

    def __init__(
        self,
        model: FasterRCNN,
        traindataset: CustomDataset,
        testdataset: CustomDataset,
        optimizer: Optimizer,
        name: str,
        cuda: int = 0,
        startepoch: int = 1,
    ) -> None:
        """
        Trainer class to train models.

        Args:
            model: model to train
            traindataset: dataset to train on
            testdataset: dataset to validate model while trainings process
            optimizer: optimizer to use
            name: name of the model in save-files and tensorboard
            cuda: number of used cuda device
        """
        self.device = (
            torch.device(f"cuda:{cuda}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        print(f"using {self.device}")
        print(f"Cuda is available: ", torch.cuda.is_available())
        if not torch.cuda.is_available():
            torch.zeros(1).cuda()

        self.model = model.to(self.device)
        self.optimizer = optimizer

        self.trainloader = DataLoader(
            traindataset, batch_size=1, shuffle=True, num_workers=2
        )
        self.testloader = DataLoader(
            testdataset, batch_size=1, shuffle=False, num_workers=2
        )

        self.bestavrgloss: Union[float, None] = None
        self.epoch = startepoch
        self.name = name
        self.startepoch = startepoch

        # setup tensor board
        train_log_dir = (
            f"{Path(__file__).parent.absolute()}/../../logs/runs/{self.name}"
        )
        print(f"{train_log_dir=}")
        self.writer = SummaryWriter(train_log_dir)  # type: ignore

        # self.example_image, self.example_target = testdataset[testdataset.getidx("mit_google_image_search-10918758-be4b5fa7bf3fea80823dabbe1e17e4136f0da811")]
        # self.train_example_image, self.train_example_target = traindataset[traindataset.getidx("mit_google_image_search-10918758-cdcd82db9ce0b61da60155c5c822b0be3884a2cf")]
        self.example_image, self.example_target = testdataset[0]
        self.train_example_image, self.train_example_target = traindataset[0]
        self.example_image = (self.example_image * 255).to(torch.uint8)
        self.train_example_image = (self.train_example_image * 255).to(torch.uint8)

    def save(self, name: str = "") -> None:
        """
        Save the model in models folder.

        Args:
            name: name of the model
        """
        os.makedirs(
            f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/",
            exist_ok=True,
        )
        torch.save(
            self.model.state_dict(),
            f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/{name}",
        )

    def load(self, name: str = "") -> None:
        """
        Load the given model.

        Args:
            name: name of the model
        """
        self.model.load_state_dict(
            torch.load(f"{Path(__file__).parent.absolute()}/../../models/{name}.pt")
        )

    def train(self, epoch: int) -> None:
        """
        Train model for given number of epochs.

        Args:
            epoch: number of epochs
        """
        for i in range(self.startepoch, self.startepoch + epoch):
            self.epoch = i
            print(f"start epoch {self.epoch}:")
            self.train_epoch()
            avgloss = self.valid()

            # early stopping
            if self.bestavrgloss is None or self.bestavrgloss > avgloss:
                self.bestavrgloss = avgloss
                self.save(f"{self.name}_es.pt")

        # save model after training
        self.save(f"{self.name}_end.pt")

    def train_epoch(self) -> None:
        """Trains one epoch."""
        loss_lst = []
        loss_classifier_lst = []
        loss_box_reg_lst = []
        loss_objectness_lst = []
        loss_rpn_box_reg_lst = []

        for img, target in tqdm(self.trainloader, desc="training"):
            img = img.to(self.device)
            target["boxes"] = target["boxes"][0].to(self.device)
            target["labels"] = target["labels"][0].to(self.device)

            # print(img.shape)

            self.optimizer.zero_grad()
            output = self.model([img[0]], [target])
            loss = sum(v for v in output.values())
            loss.backward()
            self.optimizer.step()

            loss_lst.append(loss.detach().cpu().item())
            loss_classifier_lst.append(output["loss_classifier"].detach().cpu().item())
            loss_box_reg_lst.append(output["loss_box_reg"].detach().cpu().item())
            loss_objectness_lst.append(output["loss_objectness"].detach().cpu().item())
            loss_rpn_box_reg_lst.append(
                output["loss_rpn_box_reg"].detach().cpu().item()
            )

            del img, target, output, loss

        # logging
        self.writer.add_scalar(
            "Training/loss", np.mean(loss_lst), global_step=self.epoch
        )  # type: ignore
        self.writer.add_scalar(
            "Training/loss_classifier",
            np.mean(loss_classifier_lst),
            global_step=self.epoch,
        )  # type: ignore
        self.writer.add_scalar(
            "Training/loss_box_reg", np.mean(loss_box_reg_lst), global_step=self.epoch
        )  # type: ignore
        self.writer.add_scalar(
            "Training/loss_objectness",
            np.mean(loss_objectness_lst),
            global_step=self.epoch,
        )  # type: ignore
        self.writer.add_scalar(
            "Training/loss_rpn_box_reg",
            np.mean(loss_rpn_box_reg_lst),
            global_step=self.epoch,
        )  # type: ignore
        self.writer.flush()  # type: ignore

        del (
            loss_lst,
            loss_classifier_lst,
            loss_box_reg_lst,
            loss_objectness_lst,
            loss_rpn_box_reg_lst,
        )

    def valid(self) -> float:
        """
        Validates current model on validation set.

        Returns:
            current loss
        """
        loss = []
        loss_classifier = []
        loss_box_reg = []
        loss_objectness = []
        loss_rpn_box_reg = []

        for img, target in tqdm(self.testloader, desc="validation"):
            img = img.to(self.device)
            target["boxes"] = target["boxes"][0].to(self.device)
            target["labels"] = target["labels"][0].to(self.device)

            try:
                output = self.model([img[0]], [target])
            except AssertionError:
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb)  # Fixed format
                tb_info = traceback.extract_tb(tb)
                filename, line, func, text = tb_info[-1]
                print("An error occurred on line {} in statement {}".format(line, text))
                print(target["img_number"])
                exit(1)

            loss.append(sum(v for v in output.values()).cpu().detach())
            loss_classifier.append(output["loss_classifier"].detach().cpu().item())
            loss_box_reg.append(output["loss_box_reg"].detach().cpu().item())
            loss_objectness.append(output["loss_objectness"].detach().cpu().item())
            loss_rpn_box_reg.append(output["loss_rpn_box_reg"].detach().cpu().item())

            del img, target, output

        meanloss = np.mean(loss)

        # logging
        self.writer.add_scalar(
            "Valid/loss", meanloss, global_step=self.epoch
        )  # type: ignore
        self.writer.add_scalar(
            "Valid/loss_classifier", np.mean(loss_classifier), global_step=self.epoch
        )  # type: ignore
        self.writer.add_scalar(
            "Valid/loss_box_reg", np.mean(loss_box_reg), global_step=self.epoch
        )  # type: ignore
        self.writer.add_scalar(
            "Valid/loss_objectness", np.mean(loss_objectness), global_step=self.epoch
        )  # type: ignore
        self.writer.add_scalar(
            "Valid/loss_rpn_box_reg", np.mean(loss_rpn_box_reg), global_step=self.epoch
        )  # type: ignore
        self.writer.flush()  # type: ignore

        self.model.eval()

        # predict example form training set
        pred = self.model([self.train_example_image.to(self.device) / 255])
        boxes = {
            "ground truth": self.train_example_target["boxes"],
            "prediction": pred[0]["boxes"].detach().cpu(),
        }
        colors = ["green" for i in range(boxes["prediction"].shape[0])] + [
            "red" for i in range(boxes["ground truth"].shape[0])
        ]
        labels = ["Pred" for i in range(boxes["prediction"].shape[0])] + [
            "Ground" for i in range(boxes["ground truth"].shape[0])
        ]
        result = draw_bounding_boxes(
            image=self.train_example_image,
            boxes=torch.vstack((boxes["prediction"], boxes["ground truth"])),
            colors=colors,
            labels=labels,
        )

        self.writer.add_image(
            "Training/example", result[:, ::2, ::2], global_step=self.epoch
        )  # type: ignore

        # predict example form validation set
        pred = self.model([self.example_image.to(self.device) / 255])
        boxes = {
            "ground truth": self.example_target["boxes"],
            "prediction": pred[0]["boxes"].detach().cpu(),
        }
        colors = ["green" for i in range(boxes["prediction"].shape[0])] + [
            "red" for i in range(boxes["ground truth"].shape[0])
        ]
        labels = ["Pred" for i in range(boxes["prediction"].shape[0])] + [
            "Ground" for i in range(boxes["ground truth"].shape[0])
        ]
        result = draw_bounding_boxes(
            image=self.example_image,
            boxes=torch.vstack((boxes["prediction"], boxes["ground truth"])),
            colors=colors,
            labels=labels,
        )
        self.writer.add_image(
            "Valid/example", result[:, ::2, ::2], global_step=self.epoch
        )  # type: ignore

        self.model.train()

        return meanloss


def get_model(objective: str, load_weights: Optional[str] = None) -> FasterRCNN:
    """
    Creates a FasterRCNN model for training, using the specified objective parameter.

    Args:
        objective: objective of the model (should be 'tables', 'cell', 'row' or 'col')
        load_weights: name of the model to load

    Returns:
        FasterRCNN model
    """
    params = {
        "fullimage": {"box_detections_per_img": 200},
    }

    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, **params[objective]
    )

    if load_weights:
        model.load_state_dict(
            torch.load(
                f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/"
                f"{load_weights}.pt"
            )
        )

    return model


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

    parser.add_argument(
        "--startepoch", "-se", type=int, default=1, help="number of starting epoch"
    )

    # parser.add_argument('--augmentations', "-a", action=argparse.BooleanOptionalAction)
    parser.add_argument("--augmentations", action="store_true")
    parser.add_argument(
        "--no-augmentations", dest="augmentations", action="store_false"
    )
    parser.set_defaults(augmentations=False)

    parser.add_argument("--valid", action="store_true")
    parser.add_argument("--no-valid", dest="valid", action="store_false")
    parser.set_defaults(valid=True)

    return parser.parse_args()


if __name__ == "__main__":
    from torchvision import transforms

    args = get_args()

    # check args
    if args.name == "model":
        raise ValueError("Please enter a valid model name!")

    if args.objective not in ["fullimage"]:
        raise ValueError("Please enter a valid objective must be 'fullimage'!")

    if args.dataset not in ["BonnData", "GloSat", "Tablesinthewild", "GloSAT"]:
        raise ValueError(
            "Please enter a valid dataset must be 'BonnData' or 'GloSAT' or 'Tablesinthewild'!"
        )

    if args.epochs <= 0:
        raise ValueError("Please enter a valid number of epochs must be >= 0!")

    print("start training:")
    print(f"\tname: {args.name}")
    print(f"\tobjective: {args.objective}")
    print(f"\tdataset: {args.dataset}")
    print(f"\tepochs: {args.epochs}")
    print(f"\tload: {args.load}\n")

    name = (
        f"{args.name}_{args.dataset}_{args.objective}"
        f"{'_aug' if args.augmentations else ''}_e{args.epochs}"
    )
    model = get_model(args.objective, load_weights=args.load)

    transform = None
    if args.augmentations:
        transform = torch.nn.Sequential(
            transforms.RandomApply(
                torch.nn.ModuleList(
                    [transforms.ColorJitter(brightness=(0.5, 1.5), saturation=(0, 2))]
                ),
                p=0.1,
            ),
            transforms.RandomApply(
                torch.nn.ModuleList(
                    [transforms.GaussianBlur(kernel_size=9, sigma=(2, 10))]
                ),
                p=0.1,
            ),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
            transforms.RandomGrayscale(p=0.1),
        )

    traindataset = CustomDataset(
        f"{Path(__file__).parent.absolute()}/../../data/{args.dataset}/train",
        args.objective,
        transforms=transform,
    )

    print(f"Use Valid Set: {args.valid}")

    if args.valid:
        validdataset = CustomDataset(
            f"{Path(__file__).parent.absolute()}/../../data/{args.dataset}/valid",
            args.objective,
        )
    else:
        validdataset = CustomDataset(
            f"{Path(__file__).parent.absolute()}/../../data/{args.dataset}/test",
            args.objective,
        )

    print(f"{len(traindataset)=}")
    print(f"{len(validdataset)=}")

    optimizer = AdamW(model.parameters(), lr=LR)

    trainer = Trainer(
        model, traindataset, validdataset, optimizer, name, startepoch=args.startepoch
    )
    trainer.train(args.epochs)
