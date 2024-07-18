import os
import shutil
from logging import warning
from pathlib import Path
import glob
import json

import torch
from bs4 import BeautifulSoup
from tqdm import tqdm
import warnings

import re


def getlabelname(label: str):
    """
    Ensure Labels are consistent between Melissa Dell and Chronicling Germany Newspaper Datasets
    Args:
        label: the label found in ground truth

    Returns: final label used (str)

    """
    if label.casefold() in ["paragraph", "header", "table", "caption", "heading", "advertisement", "image",
                            "inverted_text"]:
        #if label.casefold()=="advertisment":
        #    print(label)
        return label.casefold()
    elif label.casefold() == "headline":
        return "heading"
    elif label.casefold() == "cartoon/ad":
        return "advertisement"
    elif label.casefold() == "author":
        return "author"
    elif label.casefold() == "article":
        return "paragraph"
    elif re.compile("^separator_.+").match(label.casefold()):
        #print(label)
        return label.casefold()
    else:
        warnings.warn(f"Unknown label found!")
        print(label)
        return f"unknown label: {label}"


def processdata_gernews_donut(
        targetloc: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData/Zeitungen/annotations/Test",
        saveloc: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData/Zeitungen/donut/test"):
    """
    was tun bei annoncenseiten?????
    Args:
        targetloc:
        saveloc:

    Returns:

    """
    data = glob.glob(f"{targetloc}/*xml")
    os.makedirs(saveloc, exist_ok=True)
    #print(data)
    dicts = []
    for d in tqdm(data):
        with open(d) as file:
            xml = file.read()
        xmlsoup = BeautifulSoup(xml, "xml")
        #print(xmlsoup.contents)
        #return
        #print(d)
        #\S+Region
        dict = {"file_name": str(d.split('/')[-1].split('.')[-2]), "ground_truth": {
            "gt_parse": {
                "newspaper": []}}}
        for x in xmlsoup.find_all(re.compile("^\S+Region")):  #["TextRegion", "SeparatorRegion"]):
            #print(x)
            if x.has_attr('type'):
                #print(uni.get_text() for uni in x.find_all('Unicode'))
                #for uni in x.find_all('Unicode'):
                #print(uni.get_text())
                dict["ground_truth"]["gt_parse"]["newspaper"].append(
                    {getlabelname(str(x['type'])): str("\n".join([uni.get_text() for uni in x.find_all('Unicode')]))})
            elif x.Unicode:
                #for uni in x.find_all('Unicode'):
                #    print(uni.get_text())
                #print("\n".join([uni.get_text() for uni in x.find_all('Unicode')]))
                #print(x.custom, x)
                dict["ground_truth"]["gt_parse"]["newspaper"].append(
                    {getlabelname(str(x["custom"].split("structure {type:")[-1].split(";}")[-2])): str(
                        "\n".join([uni.get_text() for uni in x.find_all('Unicode')]))})
            else:
                #print(x.Unicode)
                dict["ground_truth"]["gt_parse"]["newspaper"].append(
                    {getlabelname(str(x["custom"].split("structure {type:")[-1].split(";}")[-2])): "\n"})
                #print(x["custom"].split("structure {type:")[-1].split(";}")[-2])
        dicts.append(dict)
    with open(f"{saveloc}/groundtruth.jsonl", "w") as file:
        for dict in tqdm(dicts):
            #print(dict, type(dict))
            #cdict = json.dumps(dict)
            #print([[type(key), key, type(value), value] for key, value in dict.items()])
            #print(json.dumps({str(key): str(value) for key, value in dict.items()}))
            file.write(f"{json.dumps(dict, indent=None)}\n")


def processdata_engnews_donut(targetloc: str = f"{Path(__file__).parent.absolute()}/../../data/EngNewspaper/raw",
                              saveloc: str = f"{Path(__file__).parent.absolute()}/../../data/EngNewspaper/donut"):
    """

    Args:
        targetloc:
        saveloc:

    Returns:

    """
    data = glob.glob(f"{targetloc}/*.json")
    os.makedirs(saveloc, exist_ok=True)
    dicts = []
    for datapath in tqdm(data):
        with open(datapath) as d:
            string = json.load(d)
            #print({"gt_parse": {"newspaper": [{str(k['class']): str(k['raw_text'])} for k in string['bboxes']]}})
            #dict = {"file_name": {datapath.split('/')[-1].split('.')[-2]}, "ground_truth": {"gt_parse": {[{k['class']: k['raw_text']} for k in string['bboxes']]}}}
            #dicts.append({"file_name": str(datapath.split('/')[-1].split('.')[-2]), "ground_truth": {
            #    "gt_parse": {"newspaper": [{str(k['class']): str(k['raw_text'])} for k in string['bboxes'] if
            #                               k['raw_text'] != ""]}}})
            dicts.append({"file_name": str(datapath.split('/')[-1].split('.')[-2]), "ground_truth": {
                "gt_parse": {
                    "newspaper": [{getlabelname(str(k['class'])): str(k['raw_text'])} for k in string['bboxes'] if
                                  k['raw_text'] != ""]}}})
    with open(f"{saveloc}/groundtruth.jsonl", "w") as file:
        for dict in tqdm(dicts):
            #print(dict, type(dict))
            #cdict = json.dumps(dict)
            #print([[type(key), key, type(value), value] for key, value in dict.items()])
            #print(json.dumps({str(key): str(value) for key, value in dict.items()}))
            file.write(f"{json.dumps(dict, indent=None)}\n")


def processdata_engnews_kosmos(targetloc: str = f"{Path(__file__).parent.absolute()}/../../data/EngNewspaper/raw",
                               saveloc: str = f"{Path(__file__).parent.absolute()}/../../data/EngNewspaper/Kosmos"):
    """
    Change Melissa Dell Newspaper Data to Kosmos Output Format for better comparability
    Args:
        targetloc(str): newspaper data location
        saveloc(str): new save location

    Returns:

    """
    data = glob.glob(f"{targetloc}/*.json")
    os.makedirs(saveloc, exist_ok=True)
    for datapath in tqdm(data):
        with open(datapath) as d:
            string = json.load(d)

            dict = {'width': string['scan']['width'], 'height': string['scan']['height'],
                    'results': [{'text': k['raw_text'], 'bounding box': k['bbox']} for k in string['bboxes']]}
            if len(dict['results']) > 0:
                #print(dict['results'][5])
                identifier = datapath.split('/')[-1].split('.')[-2]
                #print(identifier)
                currentjson = json.dumps(dict, indent=4)
                with open(f"{saveloc}/{identifier}.json", "w") as out:
                    out.write(currentjson)


def processdata_wildtable_inner(datapath) -> torch.Tensor:
    """process data to pt bbox file used in project group"""
    with open(datapath) as d:
        xml = BeautifulSoup(d, "xml")
    #print(xml)
    bboxes = []
    for box in xml.find_all("bndbox"):
        #print(box)
        new = torch.tensor(
            [int(float(box.xmin.get_text())), int(float(box.ymin.get_text())), int(float(box.xmax.get_text())),
             int(float(box.ymax.get_text()))], dtype=torch.float)
        #print(new)
        if new[0] < new[2] and new[1] < new[3]:
            bboxes.append(new)
        else:
            warnings.warn("invalid bbox found")
            print(datapath)
    #print(bboxes)
    if not bboxes:
        warnings.warn("no bboxes on img")
        print(datapath)
    return torch.vstack(bboxes) if bboxes else torch.empty(0, 4)


def processdata_wildtable_outer(
        datapath: str = f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/rawdata/train"):
    xmlfolder = f"{datapath}/xml"
    imfolder = f"{datapath}/images"
    target = f"{datapath}/../../train"
    for xml in tqdm(glob.glob(f"{xmlfolder}/*xml")):
        bboxfile = processdata_wildtable_inner(xml)
        impath = glob.glob(f"{imfolder}/{xml.split('/')[-1].split('.')[-3]}*")[0]
        tarfolder = f"{target}/{xml.split('/')[-1].split('.')[-3]}"
        #print(xml)
        #print(f"{tarfolder}/{impath.split('/')[-1]}")
        #print(f"{tarfolder}/{xml.split('/')[-1].split('.')[-3]}.pt")
        if bboxfile.numel():
            os.makedirs(tarfolder, exist_ok=True)
            shutil.copy(impath, dst=f"{tarfolder}/{impath.split('/')[-1]}")
            torch.save(bboxfile, f"{tarfolder}/{xml.split('/')[-1].split('.')[-3]}.pt")
        else:
            warnings.warn("empty bbox, image not added to preprocessed training data")


if __name__ == '__main__':
    processdata_wildtable_outer()
    #processdata_wildtable_inner(f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/test/test-xml-revise/test-xml-revise/0hZg6EpcTdKXk6j44umj3gAAACMAAQED.xml")
    pass
    #processdata_engnews_kosmos()
    #processdata_engnews_donut()
    #processdata_gernews_donut()
