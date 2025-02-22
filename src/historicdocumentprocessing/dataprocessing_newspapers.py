"""Preprocessing for newspaper experiments."""

import glob
import json
import os
import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup
from tqdm import tqdm


def getlabelname(label: str):
    """
    Ensure Labels are consistent between Melissa Dell and Chronicling Germany Newspaper Datasets.
    Args:
        label: the label found in ground truth

    Returns: final label used (str)

    """
    if label.casefold() in [
        "paragraph",
        "header",
        "table",
        "caption",
        "heading",
        "advertisement",
        "image",
        "inverted_text",
    ]:
        # if label.casefold()=="advertisment":
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
        # print(label)
        return label.casefold()
    else:
        warnings.warn(f"Unknown label found!")
        print(label)
        return f"unknown label: {label}"


def processdata_gernews_donut(
    targetloc: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData/Zeitungen/annotations/Test",
    saveloc: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData/Zeitungen/donut/test",
):
    """
    was tun bei annoncenseiten?????
    Args:
        targetloc:
        saveloc:

    Returns:

    """
    data = glob.glob(f"{targetloc}/*xml")
    os.makedirs(saveloc, exist_ok=True)
    # print(data)
    dicts = []
    for d in tqdm(data):
        with open(d) as file:
            xml = file.read()
        xmlsoup = BeautifulSoup(xml, "xml")
        # print(xmlsoup.contents)
        # return
        # print(d)
        # \S+Region
        dict = {
            "file_name": str(d.split("/")[-1].split(".")[-2]),
            "ground_truth": {"gt_parse": {"newspaper": []}},
        }
        for x in xmlsoup.find_all(
            re.compile("^\S+Region")
        ):  # ["TextRegion", "SeparatorRegion"]):
            # print(x)
            if x.has_attr("type"):
                # print(uni.get_text() for uni in x.find_all('Unicode'))
                # for uni in x.find_all('Unicode'):
                # print(uni.get_text())
                dict["ground_truth"]["gt_parse"]["newspaper"].append(
                    {
                        getlabelname(str(x["type"])): str(
                            "\n".join([uni.get_text() for uni in x.find_all("Unicode")])
                        )
                    }
                )
            elif x.Unicode:
                # for uni in x.find_all('Unicode'):
                #    print(uni.get_text())
                # print("\n".join([uni.get_text() for uni in x.find_all('Unicode')]))
                # print(x.custom, x)
                dict["ground_truth"]["gt_parse"]["newspaper"].append(
                    {
                        getlabelname(
                            str(
                                x["custom"]
                                .split("structure {type:")[-1]
                                .split(";}")[-2]
                            )
                        ): str(
                            "\n".join([uni.get_text() for uni in x.find_all("Unicode")])
                        )
                    }
                )
            else:
                # print(x.Unicode)
                dict["ground_truth"]["gt_parse"]["newspaper"].append(
                    {
                        getlabelname(
                            str(
                                x["custom"]
                                .split("structure {type:")[-1]
                                .split(";}")[-2]
                            )
                        ): "\n"
                    }
                )
                # print(x["custom"].split("structure {type:")[-1].split(";}")[-2])
        dicts.append(dict)
    with open(f"{saveloc}/groundtruth.jsonl", "w") as file:
        for dict in tqdm(dicts):
            # print(dict, type(dict))
            # cdict = json.dumps(dict)
            # print([[type(key), key, type(value), value] for key, value in dict.items()])
            # print(json.dumps({str(key): str(value) for key, value in dict.items()}))
            file.write(f"{json.dumps(dict, indent=None)}\n")


def processdata_engnews_donut(
    targetloc: str = f"{Path(__file__).parent.absolute()}/../../data/EngNewspaper/raw",
    saveloc: str = f"{Path(__file__).parent.absolute()}/../../data/EngNewspaper/donut",
):
    """
    Parse Melissa Dell newspaper groundtruth to donut format.
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
            # print({"gt_parse": {"newspaper": [{str(k['class']): str(k['raw_text'])} for k in string['bboxes']]}})
            # dict = {"file_name": {datapath.split('/')[-1].split('.')[-2]}, "ground_truth": {"gt_parse": {[{k['class']: k['raw_text']} for k in string['bboxes']]}}}
            # dicts.append({"file_name": str(datapath.split('/')[-1].split('.')[-2]), "ground_truth": {
            #    "gt_parse": {"newspaper": [{str(k['class']): str(k['raw_text'])} for k in string['bboxes'] if
            #                               k['raw_text'] != ""]}}})
            dicts.append(
                {
                    "file_name": str(datapath.split("/")[-1].split(".")[-2]),
                    "ground_truth": {
                        "gt_parse": {
                            "newspaper": [
                                {getlabelname(str(k["class"])): str(k["raw_text"])}
                                for k in string["bboxes"]
                                if k["raw_text"] != ""
                            ]
                        }
                    },
                }
            )
    with open(f"{saveloc}/groundtruth.jsonl", "w") as file:
        for dict in tqdm(dicts):
            # print(dict, type(dict))
            # cdict = json.dumps(dict)
            # print([[type(key), key, type(value), value] for key, value in dict.items()])
            # print(json.dumps({str(key): str(value) for key, value in dict.items()}))
            file.write(f"{json.dumps(dict, indent=None)}\n")


def processdata_engnews_kosmos(
    targetloc: str = f"{Path(__file__).parent.absolute()}/../../data/EngNewspaper/raw",
    saveloc: str = f"{Path(__file__).parent.absolute()}/../../data/EngNewspaper/Kosmos",
):
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

            dict = {
                "width": string["scan"]["width"],
                "height": string["scan"]["height"],
                "results": [
                    {"text": k["raw_text"], "bounding box": k["bbox"]}
                    for k in string["bboxes"]
                ],
            }
            if len(dict["results"]) > 0:
                # print(dict['results'][5])
                identifier = datapath.split("/")[-1].split(".")[-2]
                # print(identifier)
                currentjson = json.dumps(dict, indent=4)
                with open(f"{saveloc}/{identifier}.json", "w") as out:
                    out.write(currentjson)
