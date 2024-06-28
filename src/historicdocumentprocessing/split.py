import glob
import json
import os
import random
import shutil
from pathlib import Path
from typing import List, Dict

import pandas as pd
import torch
from tqdm import tqdm
from dataprocessing import processdata_wildtable_kosmos

def subclasssplitwildtables(impath: str = f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/test/images", xmlpath: str =  f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/test/test-xml-revise/test-xml-revise", txtfolder:str = f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/test/sub_classes"):
    """split wildtables into subclasses (TO ADD: Preprocessing for xml files)"""
    txts = glob.glob(f"{txtfolder}/*txt")
    for txt in txts:
        with open(txt) as f:
            foldername = txt.split("/")[-1].split(".")[-2]
            #print(foldername)
            destfolder = f"{'/'.join(impath.split('/')[:-2])}/preprocessed/{foldername}"
            #print(destfolder)
            for line in tqdm(f):
                #print(line)
                #print(f"{impath}/{line}")

                im = glob.glob(f"{impath}/{line.rstrip()}")[0]
                xml = glob.glob(f"{xmlpath}/{line.split('.')[-2]}.xml")[0]
                processedpt = processdata_wildtable_kosmos(xml)
                #print(processedpt)
                imfolder = f"{destfolder}/{line.split('.')[-2]}"
                os.makedirs(imfolder, exist_ok=True)
                #print(f"{imfolder}/{line.split('.')[-2]}.xml",  f"{imfolder}/{line.rstrip()}")
                shutil.copy(im, f"{imfolder}/{line.rstrip()}")
                torch.save(processedpt, f"{imfolder}/{line.split('.')[-2]}.pt")
                pass

def recreatetablesplit(datapath: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/preprocessed",
                       csvpath: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/run3_BonnData_cell_aug_loadrun_GloSAT_cell_aug_e250_es_e250_es.csv"):
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
    #print(len(imnames))
    #print(imnames)
    for i, imname in tqdm(enumerate(imnames)):
        if isinstance(imname, float):
            imname = int(imname)
        imfolder = glob.glob(f"{datapath}/{str(imname)}")
        saveloc = f"{savelocs}/{imname}"
        dest = shutil.copytree(src=imfolder[0], dst=saveloc, dirs_exist_ok=True)


def recreatenewspapersplit(datapath: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData/Zeitungen",
                           jsonpath: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData/Zeitungen/split.json"):
    """
    Recreate the Split of the Bonn Chronicling Germany Newspaper Dataset from the json file
    Args:
        datapath: path to preprocessed Bonn Table Data
        jsonpath: path to json split file

    Returns:
    """
    with open(jsonpath) as file:
        imnames = json.load(file)
    #imnames = pd.read_csv(csvpath)["image_number"][1:-1]
    for category in imnames.keys():
        saveloc = f"{datapath}/{category}"
        annotsavefolder = f"{datapath}/annotations/{category}"
        os.makedirs(annotsavefolder, exist_ok=True)
        os.makedirs(saveloc, exist_ok=True)
        #print(len(imnames))
        for imname in tqdm(imnames[category]):
            impath = glob.glob(f"{datapath}/images/{str(imname)}*")
            annotpath = glob.glob(f"{datapath}/annotations/{str(imname)}*")
            imsaveloc = f"{saveloc}/{imname}.jpg"
            annotsaveloc = f"{annotsavefolder}/{imname}.xml"
            #print(imsaveloc, annotsaveloc)
            dest = shutil.copy(src=impath[0], dst=imsaveloc)
            annotdest = shutil.copy(src=annotpath[0], dst=annotsaveloc)
        #print(i)


def newsplit(datapath: str = f"{Path(__file__).parent.absolute()}/../../data/EngNewspaper/unsorted",
             savepath: str = f"{Path(__file__).parent.absolute()}/../../data/EngNewspaper/donut",
             dataratio: List[float] = [0.8, 0.1, 0.1]):
    """
    Create New Data Split for Data in Folder
            Args:
            savepath:
            dataratio:
            datapath: Path to Data

        Returns:
        """
    #print(len(dataratio) == 3)
    assert (len(dataratio) == 3) and (dataratio[0] + dataratio[1] + dataratio[2] == 1)
    imnames = glob.glob(f"{datapath}/*jpg")
    imnum = len(imnames)
    random.shuffle(imnames)
    train = imnames[:round(imnum * dataratio[0])]
    test = imnames[round(imnum * dataratio[0]):round(imnum * (dataratio[1] + dataratio[0]))]
    valid = imnames[round(imnum * (dataratio[1] + dataratio[0])):]
    #print(len(train), len(test), len(valid), imnum)
    assert len(train) + len(test) + len(valid) == imnum
    dict = {"train": [t.split("/")[-1] for t in train], "test": [t.split("/")[-1] for t in test],
            "validation": [v.split("/")[-1] for v in valid]}
    os.makedirs(savepath, exist_ok=True)
    with open(f"{savepath}/split.json", "w") as file:
        json.dump(dict, file, indent=2)
    for cats, names in ((train, "train"), (test, "test"), (valid, "validation")):
        saveloc = f"{savepath}/{names}"
        os.makedirs(saveloc, exist_ok=True)
        for im in cats:
            dest = shutil.copy(src=im, dst=f"{saveloc}/{im.split('/')[-1]}")


if __name__ == '__main__':
    #recreatetablesplit(datapath=f"{Path(__file__).parent.absolute()}/../../data/GloSat/preprocessed",
    #                   csvpath=f"{Path(__file__).parent.absolute()}/../../data/GloSat/run_GloSAT_cell_aug_e250_es.csv")
    #recreatetablesplit()
    #recreatenewspapersplit()
    #newsplit()
    subclasssplitwildtables()
