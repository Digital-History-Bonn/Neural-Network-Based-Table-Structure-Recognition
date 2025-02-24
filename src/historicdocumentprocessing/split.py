import glob
import json
import os
import random
import shutil
import warnings
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from PIL import Image
from torchvision.io import read_image
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

from src.historicdocumentprocessing.dataprocessing import (
    processdata_wildtable_inner,
    processdata_wildtable_rowcoll,
    processdata_wildtable_tablerelative,
)


def validsplit(path: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData"):
    total = glob.glob(f"{path}/preprocessed/*")
    test = glob.glob(f"{path}/test/*")
    validlen = len(test)
    newtotal = [t for t in total if f"{path}/test/{t.split('/')[-1]}" not in test]
    # print(len(newtotal), len(test), len(total))
    assert len(newtotal) + len(test) == len(total)
    random.shuffle(newtotal)
    valid = newtotal[:validlen]
    train = newtotal[validlen:]
    # print(train)
    assert len(valid) + len(train) == len(newtotal)
    for cats, names in ((train, "train"), (valid, "valid")):
        os.makedirs(f"{path}/{names}")
        for im in cats:
            shutil.copytree(src=im, dst=f"{path}/{names}/{im.split('/')[-1]}")


def wildtablesvalidsplit(
    path: str = f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/train",
    ratio: List[float] = [0.7, 0.3],
    validfile: str = None,
):
    """
    create train/valid split for tables in the wild dataset
    Args:
        path:
        ratio: ratio of split, default chosen to keep test and valid set roughly the same size, also used in
    https://conferences.computer.org/icdar/2019/pdfs/ICDAR2019-5vPIU32iQjjaLtHlc8g8pO/6MI7Y73lJOE71Gh1fhZUQh
    /1lFyKkB62yWFXMh1YoqJmC.pdf competition paper which was used as reference for train/test split in Wired Tables in
    the Wild Dataset

    Returns:

    """
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
    # print(len(valid))
    os.makedirs(dst, exist_ok=True)
    # print(dst)
    for v in valid:
        shutil.move(v, f"{dst}/{v.split('/')[-1]}")
        print(v, f"{dst}/{v.split('/')[-1]}")
        pass


def reversewildtablesvalidsplit(
    path: str = f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild",
):
    """
    reverse train/valid split for tables in the wild dataset
    Args:
        path:
    Returns:

    """
    validpath = glob.glob(f"{path}/valid/*")
    trainpath = f"{path}/train"
    dict = {"validation": [v.split("/")[-1] for v in validpath]}
    with open(f"{path}/split1.json", "w") as file:
        json.dump(dict, file, indent=2)
    # print(len(validpath))
    for v in validpath:
        # print(f"{trainpath}/{v.split('/')[-1]}")
        shutil.move(v, f"{trainpath}/{v.split('/')[-1]}")


def subclassjoinwildtables(
    path: str = f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/testsubclasses",
    dst: str = f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/test",
):
    for folder in glob.glob(f"{path}/*"):
        # print(folder, dst)
        shutil.copytree(src=folder, dst=dst, dirs_exist_ok=True)


def subclasssplitwildtables(
    impath: str = f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/rawdata/test/images",
    xmlpath: str = f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/rawdata/test/test-xml-revise/test-xml-revise",
    txtfolder: str = f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/rawdata/test/sub_classes",
    tablerelative=False,
    rowcol=False,
):
    """split wildtables test into subclasses and do preprocessing"""
    txts = glob.glob(f"{txtfolder}/*txt")
    for txt in txts:
        with open(txt) as f:
            foldername = txt.split("/")[-1].split(".")[-2]
            # print(foldername)
            destfolder = (
                f"{'/'.join(impath.split('/')[:-2])}/../testsubclasses/{foldername}"
            )
            # print(destfolder)
            for line in tqdm(f):
                # print(line)
                # print(f"{impath}/{line}")

                im = glob.glob(f"{impath}/{line.rstrip()}")[0]
                # print(f"{impath}/{line.rstrip()}")
                xml = glob.glob(f"{xmlpath}/{line.split('.')[-2]}.xml")[0]
                processedpt = processdata_wildtable_inner(xml)
                imfolder = f"{destfolder}/{line.split('.')[-2]}"
                if processedpt.numel():
                    if read_image(im).shape[0] == 3:
                        os.makedirs(imfolder, exist_ok=True)
                        # print(f"{imfolder}/{line.split('.')[-2]}.pt",  f"{imfolder}/{line.rstrip()}")
                        shutil.copy(im, f"{imfolder}/{line.rstrip()}")
                        # print(f"{imfolder}/{line.split('.')[-2]}.pt")
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
                    for n, table in enumerate(tables):
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
                    # if len(tablelist) > 1: print(impath)
                    # print(tablefile)
                    torch.save(tablefile, f"{imfolder}/{line.split('.')[-2]}_tables.pt")

                if tablerelative and read_image(im).shape[0] == 3:
                    tablelist, celllist = processdata_wildtable_tablerelative(xml)
                    if tablelist and celllist:
                        assert len(tablelist) == len(celllist)
                        for idx in range(0, len(tablelist)):
                            # print(f"{imfolder}/{line.split('.')[-2]}_cell_{idx}.pt")
                            torch.save(
                                celllist[idx],
                                f"{imfolder}/{line.split('.')[-2]}_cell_{idx}.pt",
                            )
                            img = Image.open(im)
                            tableimg = img.crop(
                                (tuple(tablelist[idx].to(int).tolist()))
                            )
                            tableimg.save(
                                f"{imfolder}/{line.split('.')[-2]}_table_{idx}.jpg"
                            )
                        torch.save(
                            torch.vstack(tablelist),
                            f"{imfolder}/{line.split('.')[-2]}_tables.pt",
                        )
                        # print(f"{imfolder}/{line.split('.')[-2]}_tables.pt")

                        # if len(tablelist)==1:
                        #    for i in range(0,len(tablelist)):
                        #        img = Image.open(im)
                        #        tableimg = img.crop((tuple(tablelist[i].to(int).tolist())))
                        #        test = draw_bounding_boxes(image=pil_to_tensor(tableimg), boxes=celllist[i])
                        #        img = Image.fromarray(test.permute(1, 2, 0).numpy())
                        #        img.save(f"{Path(__file__).parent.absolute()}/../../images/test/croptest_{i}.jpg")
                        #    return


def recreatetablesplit(
    datapath: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData/preprocessed",
    csvpath: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/run3_BonnData_cell_aug_loadrun_GloSAT_cell_aug_e250_es_e250_es.csv",
):
    """
    Recreate the Test Split of the Bonn Table Dataset from the csv result file
    Args:
        datapath: path to preprocessed Bonn Table Data
        csvpath: path to Bonn Table csv result file

    Returns:
    """
    imnames = pd.read_csv(csvpath)["image_number"][1:-1]
    savelocs = "/".join(datapath.split("/")[:-1]) + "/test"
    os.makedirs(savelocs, exist_ok=True)
    # print(len(imnames))
    # print(imnames)
    for i, imname in tqdm(enumerate(imnames)):
        if isinstance(imname, float):
            imname = int(imname)
        imfolder = glob.glob(f"{datapath}/{str(imname)}")
        saveloc = f"{savelocs}/{imname}"
        dest = shutil.copytree(src=imfolder[0], dst=saveloc, dirs_exist_ok=True)


def recreatenewspapersplit(
    datapath: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData/Zeitungen",
    jsonpath: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData/Zeitungen/split.json",
):
    """
    Recreate the Split of the Bonn Chronicling Germany Newspaper Dataset from the json file
    Args:
        datapath: path to preprocessed Bonn Table Data
        jsonpath: path to json split file

    Returns:
    """
    with open(jsonpath) as file:
        imnames = json.load(file)
    # imnames = pd.read_csv(csvpath)["image_number"][1:-1]
    for category in imnames.keys():
        saveloc = f"{datapath}/{category}"
        annotsavefolder = f"{datapath}/annotations/{category}"
        os.makedirs(annotsavefolder, exist_ok=True)
        os.makedirs(saveloc, exist_ok=True)
        # print(len(imnames))
        for imname in tqdm(imnames[category]):
            impath = glob.glob(f"{datapath}/images/{str(imname)}*")
            annotpath = glob.glob(f"{datapath}/annotations/{str(imname)}*")
            imsaveloc = f"{saveloc}/{imname}.jpg"
            annotsaveloc = f"{annotsavefolder}/{imname}.xml"
            # print(imsaveloc, annotsaveloc)
            dest = shutil.copy(src=impath[0], dst=imsaveloc)
            annotdest = shutil.copy(src=annotpath[0], dst=annotsaveloc)
        # print(i)


def newsplit(
    datapath: str = f"{Path(__file__).parent.absolute()}/../../data/EngNewspaper/unsorted",
    savepath: str = f"{Path(__file__).parent.absolute()}/../../data/EngNewspaper/donut",
    dataratio: List[float] = [0.8, 0.1, 0.1],
):
    """
    Create New Data Split for Data in Folder
            Args:
            savepath:
            dataratio:
            datapath: Path to Data

        Returns:
    """
    # print(len(dataratio) == 3)
    assert (len(dataratio) == 3) and (dataratio[0] + dataratio[1] + dataratio[2] == 1)
    imnames = glob.glob(f"{datapath}/*jpg")
    imnum = len(imnames)
    random.shuffle(imnames)
    train = imnames[: round(imnum * dataratio[0])]
    test = imnames[
        round(imnum * dataratio[0]) : round(imnum * (dataratio[1] + dataratio[0]))
    ]
    valid = imnames[round(imnum * (dataratio[1] + dataratio[0])) :]
    # print(len(train), len(test), len(valid), imnum)
    assert len(train) + len(test) + len(valid) == imnum
    dict = {
        "train": [t.split("/")[-1] for t in train],
        "test": [t.split("/")[-1] for t in test],
        "validation": [v.split("/")[-1] for v in valid],
    }
    os.makedirs(savepath, exist_ok=True)
    with open(f"{savepath}/split.json", "w") as file:
        json.dump(dict, file, indent=2)
    for cats, names in ((train, "train"), (test, "test"), (valid, "validation")):
        saveloc = f"{savepath}/{names}"
        os.makedirs(saveloc, exist_ok=True)
        for im in cats:
            dest = shutil.copy(src=im, dst=f"{saveloc}/{im.split('/')[-1]}")


if __name__ == "__main__":
    # recreatetablesplit(datapath=f"{Path(__file__).parent.absolute()}/../../data/GloSat/preprocessed",
    #                   csvpath=f"{Path(__file__).parent.absolute()}/../../data/GloSat/run_GloSAT_cell_aug_e250_es.csv")
    # recreatetablesplit()
    # recreatenewspapersplit()
    # newsplit()
    # subclasssplitwildtables(rowcol=True)
    subclassjoinwildtables()
    # wildtablesvalidsplit(validfile=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/split.json")
    # print(pd.read_json(f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/split1.json").equals(pd.read_json(f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/split.json")))
    # validsplit(f"{Path(__file__).parent.absolute()}/../../data/GloSat")
    # reversewildtablesvalidsplit()
    pass
