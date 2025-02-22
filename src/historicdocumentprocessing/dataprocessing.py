"""Functions to preprocess data from WTW Dataset to match preprocessing in
https://github.com/Digital-History-Bonn/HistorischeTabellenSemanticExtraction/blob/main/src/TableExtraction/preprocess.py.
Note key difference regarding <imname>.pt, which is bbox coordinates on full image here (instead of image saved as pytorch tensor)

"""

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
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

from src.historicdocumentprocessing.util.tablesutil import (
    getsurroundingtable,
    gettablerelativebboxes,
)


def processdata_wildtable_inner(datapath) -> torch.Tensor:
    """Extract cell BBoxes from XML to torch tensor (coordinates relative to full image).
    Args:
        datapath: path to WTW xml file

    Returns: nx4 torch tensor with stacked BBox coordinates (xmin, ymin, xmax, ymax)
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
    """
    Extract cell BBoxes from XML to torch tensor (cell BBox coordinates relative to table, table coords relative to full image).
    Args:
        datapath: path to WTW xml file

    Returns: tuple(cell,table) of nx4 torch tensors with stacked BBox coordinates (xmin, ymin, xmax, ymax)
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
    """
    Extract cell, row, col BBoxes from XML, to torch tensor (cell, row, col BBox coordinates relative to table, table coords relative to full image).
    Extraction method for row, col from https://github.com/Digital-History-Bonn/HistorischeTabellenSemanticExtraction/blob/main/src/TableExtraction/preprocess.py.
    Args:
        datapath: path to WTW xml file

    Returns: list of table dictionaries with rows, cols, cells, tablecoords as nx4 torch tensors with stacked BBox coordinates (xmin, ymin, xmax, ymax)

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
    datapath: str = f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/rawdata/train",
    tablerelative=False,
    rowcol=True,
):
    """
    Outer method for processing WTW data from XML
    Args:
        datapath: path to raw WTW test or train folder
        tablerelative: set True if tablerelative cell BBoxes are needed
        rowcol: set True of row and column BBoxes are needed (tablerelative)

    Returns:

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


if __name__ == "__main__":
    # processdata_wildtable_outer(rowcol=True)
    # tables = processdata_wildtable_rowcoll(f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/rawdata/test/test-xml-revise/test-xml-revise/0hZg6EpcTdKXk6j44umj3gAAACMAAQED.xml")
    # print(tables)
    # img = read_image(
    #    f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/rawdata/test/images/mit_google_image_search-10918758-7f5f72bb8440c9caf8b07b28ffdc54d33bd370ab.jpg")
    # tables = processdata_wildtable_rowcoll(f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/rawdata/test/test-xml-revise/test-xml-revise/mit_google_image_search-10918758-7f5f72bb8440c9caf8b07b28ffdc54d33bd370ab.xml")
    # print(tables)
    #
    # img = read_image(f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/rawdata/train/images/008ae846c799f80b34252eb58358d166aa3a82d2.jpg")
    # tables = torch.load(f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/train/008ae846c799f80b34252eb58358d166aa3a82d2/008ae846c799f80b34252eb58358d166aa3a82d2_tables.pt")
    # print(tables)
    # for n, table in enumerate(tables):
    #    row = reversetablerelativebboxes_inner(tablebbox=table, cellbboxes=torch.load(f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/train/008ae846c799f80b34252eb58358d166aa3a82d2/008ae846c799f80b34252eb58358d166aa3a82d2_col_{n}.pt"))
    #    rowimg = draw_bounding_boxes(image=img, boxes=row, colors=["black" for i in range(len(row))],
    #                                                              labels=["row" for i in range(len(row))])
    #    rowimg = Image.fromarray(rowimg.permute(1, 2, 0).numpy())
    #    rowimg.save(f"{Path(__file__).parent.absolute()}/../../images/coltest_saved_{n}.jpg")
    # row = reversetablerelativebboxes_inner(tablebbox=table[0], cellbboxes=torch.load(f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/train/000e84efe82c0f7c17c77334704e93bc/000e84efe82c0f7c17c77334704e93bc_row_0.pt"))
    # rowimg = draw_bounding_boxes(image=img, boxes=row, colors=["black" for i in range(len(row))],
    #                             labels=["row" for i in range(len(row))])
    # rowimg = Image.fromarray(rowimg.permute(1, 2, 0).numpy())
    # rowimg.save(f"{Path(__file__).parent.absolute()}/../../images/rowtest_saved.jpg")
    """
    for n,table in enumerate(tables):
        tab = tables[n]["table"]
        rows = reversetablerelativebboxes_inner(tablebbox=tab, cellbboxes=tables[n]["rows"])
        colls = reversetablerelativebboxes_inner(tablebbox=tab, cellbboxes=tables[n]["cols"])
        cells = reversetablerelativebboxes_inner(tablebbox=tab, cellbboxes=tables[n]["cells"])
        #rows, colls = processdata_wildtable_rowcoll(f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/rawdata/test/test-xml-revise/test-xml-revise/0hZg6EpcTdKXk6j44umj3gAAACMAAQED.xml")
        #img = read_image(f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/rawdata/test/images/0hZg6EpcTdKXk6j44umj3gAAACMAAQED.jpg")
        cellimg = draw_bounding_boxes(image=img, boxes=cells, colors=["black" for i in range(len(cells))],
                                     labels=["cell" for i in range(len(cells))])
        cellimg = Image.fromarray(cellimg.permute(1, 2, 0).numpy())
        cellimg.save(f"{Path(__file__).parent.absolute()}/../../images/celltest_multi_{n}.jpg")
        rowimg = draw_bounding_boxes(image=img, boxes=rows, colors=["black" for i in range(len(rows))], labels=["row" for i in range(len(rows))])
        colimg = draw_bounding_boxes(image=img, boxes=colls, colors=["black" for i in range(len(colls))], labels=["col" for i in range(len(colls))])
        rowimg= Image.fromarray(rowimg.permute(1, 2, 0).numpy())
        rowimg.save(f"{Path(__file__).parent.absolute()}/../../images/rowtest_multi_{n}.jpg")
        colimg= Image.fromarray(colimg.permute(1, 2, 0).numpy())
        colimg.save(f"{Path(__file__).parent.absolute()}/../../images/coltest_multi_{n}.jpg")
    """
    # rows, colls = processdata_wildtable_rowcoll(
    #    f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/rawdata/test/test-xml-revise/test-xml-revise/0hZg6EpcTdKXk6j44umj3gAAACMAAQED.xml")
    # img = read_image(
    #    f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/rawdata/test/images/0hZg6EpcTdKXk6j44umj3gAAACMAAQED.jpg")
    # rowimg = draw_bounding_boxes(image=img, boxes=rows, colors=["black" for i in range(len(rows))])
    # colimg = draw_bounding_boxes(image=img, boxes=colls, colors=["black" for i in range(len(colls))])
    # rowimg = Image.fromarray(rowimg.permute(1, 2, 0).numpy())
    # rowimg.save(f"{Path(__file__).parent.absolute()}/../../images/rowtest.jpg")
    # colimg = Image.fromarray(colimg.permute(1, 2, 0).numpy())
    # colimg.save(f"{Path(__file__).parent.absolute()}/../../images/coltest.jpg")
    # processdata_wildtable_outer(tablerelative=True)
    # processdata_wildtable_inner(f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/test/test-xml-revise/test-xml-revise/0hZg6EpcTdKXk6j44umj3gAAACMAAQED.xml")
    pass
