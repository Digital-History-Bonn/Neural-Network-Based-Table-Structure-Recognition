"""Various functions for splitting and combining data."""

import argparse
import glob
import json
import os
import random
import shutil
import warnings
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
from PIL import Image
from torchvision.io import read_image
from tqdm import tqdm

from src.historicdocumentprocessing.dataprocessing import (
    processdata_wildtable_inner,
    processdata_wildtable_rowcoll,
    processdata_wildtable_tablerelative,
)


def validsplit(path: str):  # f"{Path(__file__).parent.absolute()}/../../data/BonnData")
    """Split train dataset into train and valid data (with valid data same size as test data).

    Args:
        path: path to dataset

    """
    total = glob.glob(f"{path}/preprocessed/*")
    test = glob.glob(f"{path}/test/*")
    validlen = len(test)
    newtotal = [t for t in total if f"{path}/test/{t.split('/')[-1]}" not in test]
    assert len(newtotal) + len(test) == len(total)
    random.shuffle(newtotal)
    valid = newtotal[:validlen]
    train = newtotal[validlen:]
    assert len(valid) + len(train) == len(newtotal)
    for cats, names in ((train, "train"), (valid, "valid")):
        os.makedirs(f"{path}/{names}")
        for im in cats:
            shutil.copytree(src=im, dst=f"{path}/{names}/{im.split('/')[-1]}")


def wildtablesvalidsplit(
    path: str,  # = f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/train",
    ratio: Optional[List[float]] = None,
    validfile: Optional[str] = None,
):
    """Create train/valid split for tables in the wild dataset.

    Args:
        path: path to preprocessed train data
        ratio: ratio of split, default chosen to keep test and valid set roughly the same size, also used in https://conferences.computer.org/icdar/2019/pdfs/ICDAR2019-5vPIU32iQjjaLtHlc8g8pO/6MI7Y73lJOE71Gh1fhZUQh/1lFyKkB62yWFXMh1YoqJmC.pdf competition paper which was used as reference for train/test split in Wired Tables in the Wild Dataset
        validfile: valid file (json)

    """
    if not ratio:
        ratio = [0.7, 0.3]
    dst = f"{'/'.join(path.split('/')[:-1])}/valid"
    trainlist = glob.glob(f"{path}/*")
    if not validfile:
        random.shuffle(trainlist)
        train = trainlist[: round(len(trainlist) * ratio[0])]
        valid = trainlist[
            round(len(trainlist) * ratio[0]) : round(
                len(trainlist) * (ratio[1] + ratio[0])
            )
        ]
        assert len(train) + len(valid) == len(trainlist)
    else:
        imnames = pd.read_json(validfile)["validation"]
        valid = [f"{path}/{im}" for im in imnames]
        print(valid)
        pass
    os.makedirs(dst, exist_ok=True)
    for v in valid:
        shutil.move(v, f"{dst}/{v.split('/')[-1]}")
        print(v, f"{dst}/{v.split('/')[-1]}")
        pass


def reversewildtablesvalidsplit(
    path: str,  # f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild"
):
    """Reverse train/valid split for tables in the wild dataset.

    Args:
        path: path to dataset

    """
    validpath = glob.glob(f"{path}/valid/*")
    trainpath = f"{path}/train"
    dict = {"validation": [v.split("/")[-1] for v in validpath]}
    with open(f"{path}/split.json", "w") as file:
        json.dump(dict, file, indent=2)
    # print(len(validpath))
    for v in validpath:
        # print(f"{trainpath}/{v.split('/')[-1]}")
        shutil.move(v, f"{trainpath}/{v.split('/')[-1]}")


def subclassjoinwildtables(
    path: str,  # f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/testsubclasses"
    dst: str,  # f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/test"
):
    """Join subclassfolders into one test folder.

    Args:
        path: path to folder with subclassfolders
        dst: destination folder

    """
    for folder in glob.glob(f"{path}/*"):
        # print(folder, dst)
        shutil.copytree(src=folder, dst=dst, dirs_exist_ok=True)


def subclasssplitwildtables(
    impath: str,  # f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/rawdata/test/images"
    xmlpath: str,  # f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/rawdata/test/test-xml-revise/test-xml-revise"
    txtfolder: str,  # f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/rawdata/test/sub_classes"
    tablerelative=False,
    rowcol=False,
):
    """Split wildtables test into subclasses and do preprocessing.

    Args:
        impath: path to images
        xmlpath: path to xml files
        txtfolder: path to folder with txt files with subclass lists
        tablerelative: wether tablerelative BBoxes are needed
        rowcol: wether row and column BBoxes are needed

    """
    txts = glob.glob(f"{txtfolder}/*txt")
    for txt in txts:
        with open(txt) as f:
            foldername = txt.split("/")[-1].split(".")[-2]
            destfolder = (
                f"{'/'.join(impath.split('/')[:-2])}/../testsubclasses/{foldername}"
            )
            for line in tqdm(f):
                im = glob.glob(f"{impath}/{line.rstrip()}")[0]
                xml = glob.glob(f"{xmlpath}/{line.split('.')[-2]}.xml")[0]
                processedpt = processdata_wildtable_inner(xml)
                imfolder = f"{destfolder}/{line.split('.')[-2]}"
                if processedpt.numel():
                    if read_image(im).shape[0] == 3:
                        os.makedirs(imfolder, exist_ok=True)
                        shutil.copy(im, f"{imfolder}/{line.rstrip()}")
                        torch.save(processedpt, f"{imfolder}/{line.split('.')[-2]}.pt")
                    else:
                        print(f"Wrong image dim at {im}")
                else:
                    warnings.warn(
                        "empty bbox, image not added to preprocessed test data"
                    )

                if (
                    rowcol
                    and processedpt.numel()
                    and (read_image(im) / 255).shape[0] == 3
                ):
                    tables = processdata_wildtable_rowcoll(xml)
                    tablelist = []
                    for n, _table in enumerate(tables):
                        tab = tables[n]["table"]
                        tablelist += [tab]
                        img = Image.open(im)
                        tableimg = img.crop(tuple(tab.to(int).tolist()))
                        tableimg.save(f"{imfolder}/{line.split('.')[-2]}_table_{n}.jpg")
                        rows = tables[n]["rows"]
                        colls = tables[n]["cols"]
                        cells = tables[n]["cells"]
                        torch.save(
                            cells,
                            f"{imfolder}/{line.split('.')[-2]}_cell_{n}.pt",
                        )
                        torch.save(
                            rows,
                            f"{imfolder}/{line.split('.')[-2]}_row_{n}.pt",
                        )
                        torch.save(
                            colls,
                            f"{imfolder}/{line.split('.')[-2]}_col_{n}.pt",
                        )
                    tablefile = torch.vstack(tablelist)
                    torch.save(tablefile, f"{imfolder}/{line.split('.')[-2]}_tables.pt")

                if tablerelative and read_image(im).shape[0] == 3:
                    tablelist, celllist = processdata_wildtable_tablerelative(xml)
                    if tablelist and celllist:
                        assert len(tablelist) == len(celllist)
                        for idx in range(0, len(tablelist)):
                            torch.save(
                                celllist[idx],
                                f"{imfolder}/{line.split('.')[-2]}_cell_{idx}.pt",
                            )
                            img = Image.open(im)
                            tableimg = img.crop(
                                (tuple(tablelist[idx].to(int).tolist()))  # type: ignore
                            )
                            tableimg.save(
                                f"{imfolder}/{line.split('.')[-2]}_table_{idx}.jpg"
                            )
                        torch.save(
                            torch.vstack(tablelist),
                            f"{imfolder}/{line.split('.')[-2]}_tables.pt",
                        )


def recreatetablesplit(
    datapath: str,  # f"{Path(__file__).parent.absolute()}/../../data/BonnData/preprocessed"
    csvpath: str,  # f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/run3_BonnData_cell_aug_loadrun_GloSAT_cell_aug_e250_es_e250_es.csv"
):
    """Recreate the Test Split of the Bonn Table Dataset from the csv result file.

    Args:
        datapath: path to preprocessed Bonn Table Data
        csvpath: path to Bonn Table csv result file

    """
    imnames = (
        pd.read_csv(csvpath)["image_number"].dropna().drop_duplicates()
    )  # because of this previous error, IMG_20190821_164100 is missing in test set of BonnData and 465 is missing in test set of GloSAT for original results
    savelocs = "/".join(datapath.split("/")[:-1]) + "/test"
    os.makedirs(savelocs, exist_ok=True)
    for _i, imname in tqdm(enumerate(imnames)):
        if isinstance(imname, float):
            imname = int(imname)
        imfolder = glob.glob(f"{datapath}/{str(imname)}")
        saveloc = f"{savelocs}/{imname}"
        shutil.copytree(src=imfolder[0], dst=saveloc, dirs_exist_ok=True)


def savetablesplit(
    datapath: str,  # f"{Path(__file__).parent.absolute()}/../../data/BonnData"
    splitname: str = "split",
):
    """Save the split of bonn table dataset or glosat dataset to file.

    Args:
        datapath: path to dataset folder
        splitname: name of split file

    """
    dict: Dict[str, Union[List[str], Dict[str, int]]] = {"len": {}}
    for split in ("train", "test", "valid"):
        folder = glob.glob(f"{datapath}/{split}/*")
        dict.update({split: [f.split("/")[-1] for f in folder]})
        dict["len"].update({split: len(folder)})  # type: ignore
    filename = f"{datapath}/{splitname}.json"
    with open(filename, "w") as file:
        json.dump(dict, file, indent=2)
        print(f"data saved to {filename}.")


def get_args() -> argparse.Namespace:
    """Define args."""  # noqa: DAR201
    parser = argparse.ArgumentParser(description="split")
    parser.add_argument(
        "--operation", choices=["recreatesplit, validsplit, subclasssplit", "savesplit"]
    )

    parser.add_argument(
        "--datasetname",
        default="BonnData",
        choices=["BonnData", "GloSat", "Tablesinthewild"],
    )
    parser.add_argument(
        "--splitfile", default="split", help="filename of file with split file data"
    )

    parser.add_argument("--tablerelative", action="store_true", default=False)
    parser.add_argument(
        "--no-tablerelative", dest="tablerelative", action="store_false"
    )

    parser.add_argument("--rowcol", action="store_true", default=False)
    parser.add_argument("--no-rowcol", dest="filter", action="store_false")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.datasetname == "tabletransformer":
        if args.operation == "subclasssplit":
            subclasssplitwildtables(
                impath="./data/Tablesinthewild/rawdata/test/images",
                xmlpath="./data/Tablesinthewild/rawdata/test/test-xml-revise/test-xml-revise",
                txtfolder="./data/Tablesinthewild/rawdata/test/sub_classes",
                tablerelative=args.tablerelative,
                rowcol=args.rowcol,
            )
        else:
            wildtablesvalidsplit(
                path="./data/Tablesinthewild/train",
                validfile=(
                    None
                    if args.splitfile == ""
                    else f"./data/Tablesinthewild/{args.splitfile}.json"
                ),
            )
    else:
        if args.operation == "validsplit":
            validsplit(path=f"./data/{args.datasetname}")
        elif args.operation == "recreatesplit":
            recreatetablesplit(
                datapath=f"./data/{args.datasetname}/preprocessed",
                csvpath=f"./data/{args.datasetname}/{args.splitfile}.csv",
            )
        elif args.operation == "savesplit":
            savetablesplit(
                datapath=f"./data/{args.datasetname}", splitname=args.splitfile
            )
            print("split saved.")
