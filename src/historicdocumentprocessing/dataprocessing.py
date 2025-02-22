"""Functions to preprocess data from WTW Dataset to match preprocessing in https://github.com/Digital-History-Bonn/HistorischeTabellenSemanticExtraction/blob/main/src/TableExtraction/preprocess.py.

Note key difference regarding <imname>.pt, which is bbox coordinates on full image here (instead of image saved as pytorch tensor)

"""
import argparse
import glob
import os
import shutil
import warnings
from pathlib import Path
from typing import Dict, List

import torch
from bs4 import BeautifulSoup
from PIL import Image
from scipy.cluster.hierarchy import DisjointSet
from torchvision.io import read_image
from tqdm import tqdm

from src.historicdocumentprocessing.util.tablesutil import (
    getsurroundingtable,
    gettablerelativebboxes,
)


def processdata_wildtable_inner(datapath) -> torch.Tensor:
    """Extract cell BBoxes from XML to torch tensor (coordinates relative to full image).

    Args:
        datapath: path to WTW xml file

    Returns:
        nx4 torch tensor with stacked BBox coordinates (xmin, ymin, xmax, ymax)
    """
    with open(datapath) as d:
        xml = BeautifulSoup(d, "xml")
    bboxes = []
    for box in xml.find_all("bndbox"):
        new = torch.tensor(
            [
                int(float(box.xmin.get_text())),
                int(float(box.ymin.get_text())),
                int(float(box.xmax.get_text())),
                int(float(box.ymax.get_text())),
            ],
            dtype=torch.float,
        )
        if new[0] < new[2] and new[1] < new[3]:
            bboxes.append(new)
        else:
            warnings.warn("invalid bbox found")
            print(datapath)
    if not bboxes:
        warnings.warn("no bboxes on img")
        print(datapath)
    return torch.vstack(bboxes) if bboxes else torch.empty(0, 4)


def processdata_wildtable_tablerelative(
    datapath: str,
) -> (List[torch.Tensor], List[torch.Tensor]):
    """Extract cell BBoxes from XML to torch tensor (cell BBox coordinates relative to table, table coords relative to full image).

    Args:
        datapath: path to WTW xml file

    Returns:
        tuple(cell,table) of nx4 torch tensors with stacked BBox coordinates (xmin, ymin, xmax, ymax)
    """
    with open(datapath) as d:
        xml = BeautifulSoup(d, "xml")
    tablelist = []
    celllist = []
    idx = 0
    bboxes = []
    for box in xml.find_all("bndbox"):
        new = torch.tensor(
            [
                int(float(box.xmin.get_text())),
                int(float(box.ymin.get_text())),
                int(float(box.xmax.get_text())),
                int(float(box.ymax.get_text())),
            ],
            dtype=torch.float,
        )
        if new[0] < new[2] and new[1] < new[3]:
            if int(box.tableid.get_text()) == idx:
                bboxes.append(new)
            else:
                tablelist.append(getsurroundingtable(torch.vstack(bboxes)))
                celllist.append(gettablerelativebboxes(torch.vstack(bboxes)))
                bboxes.clear()
                bboxes.append(new)
        else:
            warnings.warn("invalid bbox found")
            print(datapath)
    tablelist.append(getsurroundingtable(torch.vstack(bboxes)))
    celllist.append(gettablerelativebboxes(torch.vstack(bboxes)))
    if not bboxes:
        warnings.warn("no bboxes on img")
        print(datapath)
    return tablelist, celllist


def processdata_wildtable_rowcoll(
    datapath: str,
) -> (List[Dict]):
    """Extract cell, row, col BBoxes from XML, to torch tensor (cell, row, col BBox coordinates relative to table, table coords relative to full image).

    Extraction method for row, col from https://github.com/Digital-History-Bonn/HistorischeTabellenSemanticExtraction/blob/main/src/TableExtraction/preprocess.py.

    Args:
        datapath: path to WTW xml file

    Returns:
        list of table dictionaries with rows, cols, cells, tablecoords as nx4 torch tensors with stacked BBox coordinates (xmin, ymin, xmax, ymax)
    """
    with open(datapath) as d:
        xml = BeautifulSoup(d, "xml")
    tables: Dict[int, List[Dict]] = {}
    for box in xml.find_all("bndbox"):
        new = torch.tensor(
            [
                int(float(box.xmin.get_text())),
                int(float(box.ymin.get_text())),
                int(float(box.xmax.get_text())),
                int(float(box.ymax.get_text())),
            ],
            dtype=torch.float,
        )
        if new[0] < new[2] and new[1] < new[3]:
            if int(box.tableid.get_text()) in tables.keys():
                tables[int(box.tableid.get_text())] += [
                    {
                        "bbox": new,
                        "startcol": int(box.startcol.get_text()),
                        "endcol": int(box.endcol.get_text()),
                        "startrow": int(box.startrow.get_text()),
                        "endrow": int(box.endrow.get_text()),
                    }
                ]
            else:
                tables.update(
                    {
                        int(box.tableid.get_text()): [
                            {
                                "bbox": new,
                                "startcol": int(box.startcol.get_text()),
                                "endcol": int(box.endcol.get_text()),
                                "startrow": int(box.startrow.get_text()),
                                "endrow": int(box.endrow.get_text()),
                            }
                        ]
                    }
                )

    finaltables: List[Dict] = []
    for t in tables.keys():
        columns: Dict[int, torch.Tensor] = {}
        rows: Dict[int, torch.Tensor] = {}
        col_joins = []
        row_joins = []
        cells = []
        for box in tables[t]:
            new = box["bbox"]
            if new[0] < new[2] and new[1] < new[3]:
                cells += [new]
                startcol = box["startcol"]
                endcol = box["endcol"]
                startrow = box["startrow"]
                endrow = box["endrow"]
                for i in range(startcol, endcol + 1):
                    if i in columns.keys():
                        columns[i] = torch.vstack((columns[i], new))
                        col_joins += [[startcol, i]]
                    else:
                        columns.update({i: new})
                        col_joins += [[startcol, i]]
                for i in range(startrow, endrow + 1):
                    if i in rows.keys():
                        rows[i] = torch.vstack((rows[i], new))
                        row_joins += [[startrow, i]]
                    else:
                        rows.update({i: new})
                        row_joins += [[startrow, i]]
        col_set = DisjointSet(columns.keys())
        for joinop in col_joins:
            col_set.merge(joinop[0], joinop[1])
        finalcols = []
        for subset in col_set.subsets():
            finalcols += [
                getsurroundingtable(torch.vstack(([columns[t] for t in subset])))
            ]
        row_set = DisjointSet(rows.keys())
        for joinop in row_joins:
            row_set.merge(joinop[0], joinop[1])
        finalrows = []
        for subset in row_set.subsets():
            finalrows += [getsurroundingtable(torch.vstack([rows[t] for t in subset]))]
        finaltables += [
            {
                "rows": gettablerelativebboxes(torch.vstack(finalrows)),
                "cols": gettablerelativebboxes(torch.vstack(finalcols)),
                "cells": gettablerelativebboxes(torch.vstack(cells)),
                "table": getsurroundingtable(torch.vstack(cells)),
                "num": t,
            }
        ]
    return finaltables


def processdata_wildtable_outer(
    datapath: str,  # f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/rawdata/train"
    tablerelative: bool = False,
    rowcol: bool = True,
):
    """Outer method for processing WTW data from XML.

    Args:
        datapath: path to raw WTW test or train folder
        tablerelative: set True if tablerelative cell BBoxes are needed
        rowcol: set True of row and column BBoxes are needed (tablerelative)
    """
    xmlfolder = f"{datapath}/xml"
    imfolder = f"{datapath}/images"
    target = f"{datapath}/../../{datapath.split('/')[-1]}"
    for xml in tqdm(glob.glob(f"{xmlfolder}/*xml")):
        bboxfile = processdata_wildtable_inner(xml)
        impath = glob.glob(f"{imfolder}/{xml.split('/')[-1].split('.')[-3]}.jpg")[0]
        tarfolder = f"{target}/{xml.split('/')[-1].split('.')[-3]}"
        if bboxfile.numel():
            if (read_image(impath) / 255).shape[0] == 3:
                os.makedirs(tarfolder, exist_ok=True)
                shutil.copy(impath, dst=f"{tarfolder}/{impath.split('/')[-1]}")
                torch.save(
                    bboxfile, f"{tarfolder}/{xml.split('/')[-1].split('.')[-3]}.pt"
                )
            else:
                warnings.warn(
                    "image in wrong format, not added to preprocessed training data"
                )
                print(impath)
        else:
            warnings.warn("empty bbox, image not added to preprocessed training data")
        if rowcol and bboxfile.numel() and (read_image(impath) / 255).shape[0] == 3:
            tables = processdata_wildtable_rowcoll(xml)
            tablelist = []
            for n, _table in enumerate(tables):
                tab = tables[n]["table"]
                tablelist += [tab]
                img = Image.open(impath)
                tableimg = img.crop(tuple(tab.to(int).tolist()))
                tableimg.save(
                    f"{tarfolder}/{xml.split('/')[-1].split('.')[-3]}_table_{n}.jpg"
                )
                rows = tables[n]["rows"]
                colls = tables[n]["cols"]
                cells = tables[n]["cells"]
                torch.save(
                    cells,
                    f"{tarfolder}/{xml.split('/')[-1].split('.')[-3]}_cell_{n}.pt",
                )
                torch.save(
                    rows,
                    f"{tarfolder}/{xml.split('/')[-1].split('.')[-3]}_row_{n}.pt",
                )
                torch.save(
                    colls,
                    f"{tarfolder}/{xml.split('/')[-1].split('.')[-3]}_col_{n}.pt",
                )
            tablefile = torch.vstack(tablelist)
            torch.save(
                tablefile, f"{tarfolder}/{xml.split('/')[-1].split('.')[-3]}_tables.pt"
            )

        if tablerelative and (read_image(impath) / 255).shape[0] == 3:
            tablelist, celllist = processdata_wildtable_tablerelative(xml)
            if tablelist and celllist:
                assert len(tablelist) == len(celllist)
                for idx in range(0, len(tablelist)):
                    torch.save(
                        celllist[idx],
                        f"{tarfolder}/{xml.split('/')[-1].split('.')[-3]}_cell_{idx}.pt",
                    )
                    img = Image.open(impath)
                    tableimg = img.crop((tuple(tablelist[idx].to(int).tolist())))
                    tableimg.save(
                        f"{tarfolder}/{xml.split('/')[-1].split('.')[-3]}_table_{idx}.jpg"
                    )
                torch.save(
                    torch.vstack(tablelist),
                    f"{tarfolder}/{xml.split('/')[-1].split('.')[-3]}_tables.pt",
                )
            # if len(tablelist)==2:
            #    for i in range(0,len(tablelist)):
            #        img = Image.open(impath)
            #        tableimg = img.crop((tuple(tablelist[i].to(int).tolist())))
            #        test = draw_bounding_boxes(image=pil_to_tensor(tableimg), boxes=celllist[i])
            #        img = Image.fromarray(test.permute(1, 2, 0).numpy())
            #        img.save(f"{Path(__file__).parent.absolute()}/../../images/test/train_2_croptest_{i}.jpg")
            #    return


def get_args() -> argparse.Namespace:
    """Define args."""
    parser = argparse.ArgumentParser(description="dataprocessing_wtw")
    parser.add_argument('-p', '--path', default=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/rawdata")
    parser.add_argument('--folder', default="train")
    parser.add_argument('--tablerelative', action='store_true', default=False)
    parser.add_argument('--no-tablerelative', dest='tablerelative', action='store_false')
    parser.add_argument('--rowcol', action='store_true', default=False)
    parser.add_argument('--no-rowcol', dest='rowcol', action='store_false')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    path = f"{args.path}/{args.folder}"
    processdata_wildtable_outer(datapath=path, tablerelative=args.tablerelative, rowcol=args.rowcol)
