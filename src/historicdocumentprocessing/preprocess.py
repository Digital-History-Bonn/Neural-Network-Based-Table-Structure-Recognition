"""script for extracting annotations and preprocessing images."""

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from bs4 import BeautifulSoup
from PIL import Image, ImageOps
from scipy.cluster.hierarchy import DisjointSet
from torchvision import transforms
from tqdm import tqdm

from .util.preprocess_utils import convert_coords, get_bbox


def extract_annotation(
    file: str, mode: str = "maximum", table_relative: bool = True
) -> Tuple[List[Dict[str, Sequence[int]]], List[Dict[str, Tuple[int, int, int, int]]]]:
    """Extracts annotation data from transkribus xml file.

    Args:
        file: path to file
        mode: mode for bounding box extraction options are 'maximum' and 'corners'
            'maximum': creates a bounding box including all points annotated
            'corners': creates a bounding box by connecting the corners of the annotation
        table_relative: if True, bounding boxes for cell and row are calculated relative
                        to table position

    Returns:
        A Tuple of two list the first containing all tables the other all text regions
    """
    tables = []
    textregions = []

    with open(file, "r", encoding="utf-8") as file:  # type: ignore
        xml_content = file.read()  # type: ignore

    # Parse the XML content
    soup = BeautifulSoup(xml_content, "xml")

    # Find all TextLine elements and extract the Baseline points
    for textline in soup.find_all("TextRegion"):
        textcoords = textline.find("Coords")["points"]  # type: ignore
        if textcoords.find("NaN") == -1:  # type: ignore
            text = {
                "coords": get_bbox(convert_coords(textline.find("Coords")["points"]))  # type: ignore
            }
            textregions.append(text)

    for table in soup.find_all("TableRegion"):
        tablecoords = table.find("Coords")["points"] # type: ignore
        if tablecoords.find("NaN") != -1: # type: ignore
            continue

        t = {
            "coords": get_bbox(convert_coords(tablecoords)),  # type: ignore
            "cells": [],
            "columns": [],
            "rows": [],
        }

        maxcol = 0

        columns: Dict[int, List[List[int]]] = (
            {}
        )  # dictionary of columns and their points
        rows: Dict[int, List[List[int]]] = {}  # dictionary of rows and their points
        col_joins = []  # list of join operations for columns
        row_joins = []  # list of join operations for rows

        for cell in table.find_all("TableCell"):
            maxcol = max(maxcol, int(cell["col"]))  # type: ignore
        firstcell = table.find("TableCell")

        if (
            cell.get("rowSpan") is not None
            and int(firstcell["colSpan"]) == maxcol + 1  # type: ignore
            and int(firstcell["row"]) == 0  # type: ignore
        ):
            coord1 = t["coords"]
            coord2 = get_bbox(convert_coords(firstcell.find("Coords")["points"]))  # type: ignore
            t["coords"] = (coord1[0], coord2[3], coord1[2], coord1[3])

        if table_relative:
            coord = t["coords"]
        else:
            coord = None

        # iterate over cells in table
        for cell in table.find_all("TableCell"):
            # get points and corners of cell
            points = convert_coords(cell.find("Coords")["points"])  # type: ignore

            # uses corner points of cell if mode is 'corners'
            corners = (
                [int(x) for x in cell.find("CornerPts").text.split_dataset()]  # type: ignore
                if mode == "corners"
                else None
            )

            # get bounding box
            bbox = get_bbox(points, corners, coord)  # type: ignore

            # add to dictionary
            x_flat = bbox[0] >= bbox[2]  # bbox flatt in x dim
            y_flat = bbox[1] >= bbox[3]  # bbox flatt in y dim

            # check if Cell is a header for table
            no_header_cell = cell.get("rowSpan") is not None and not (
                int(cell["colSpan"]) == maxcol + 1 and int(cell["row"]) == 0  # type: ignore
            )

            if not x_flat and not y_flat and no_header_cell:
                t["cells"].append(bbox)  # type: ignore

                # calc rows
                # add row number to dict
                if int(cell["row"]) in rows.keys():  # type: ignore
                    rows[int(cell["row"])].extend(points.tolist())  # type: ignore
                else:
                    rows[int(cell["row"])] = points.tolist() # type: ignore

                # when cell over multiple rows create a join operation
                if int(cell["rowSpan"]) > 1:  # type: ignore
                    row_joins.extend(
                        [
                            (int(cell["row"]), int(cell["row"]) + s)  # type: ignore
                            for s in range(1, int(cell["rowSpan"]))  # type: ignore
                        ]
                    )

                # add col number to dict
                if int(cell["col"]) in columns.keys():  # type: ignore
                    columns[int(cell["col"])].extend(points.tolist())  # type: ignore
                else:
                    columns[int(cell["col"])] = points.tolist()  # type: ignore

                # when cell over multiple columns create a join operation
                if int(cell["colSpan"]) > 1:  # type: ignore
                    col_joins.extend(
                        [
                            (int(cell["col"]), int(cell["col"]) + s)  # type: ignore
                            for s in range(1, int(cell["colSpan"]))  # type: ignore
                        ]
                    )

        # join overlapping rows
        row_set = DisjointSet(rows.keys())
        for join in row_joins:
            if join[0] in rows.keys() and join[1] in rows.keys():
                row_set.merge(*join)

        rows_list = [
            [point for key in lst for point in rows[key]] for lst in row_set.subsets()
        ]
        t["rows"] = [get_bbox(np.array(col), tablebbox=coord) for col in rows_list]  # type: ignore

        # join overlapping columns
        col_set = DisjointSet(columns.keys())
        for join in col_joins:
            if join[0] in columns.keys() and join[1] in columns.keys():
                col_set.merge(*join)

        cols_list = [
            [point for key in lst for point in columns[key]]
            for lst in col_set.subsets()
        ]
        t["columns"] = [
            get_bbox(np.array(col), tablebbox=coord) for col in cols_list  # type: ignore
        ]  # type: ignore

        if t["columns"] and t["rows"]:
            tables.append(t)

    return tables, textregions


def preprocess(
    image: str,
    tables: List[Dict[str, Sequence[int]]],
    target: str,
    file_name: str,
    text: Optional[List[Dict[str, Tuple[int, int, int, int]]]] = None,
) -> None:
    """Preprocessing.

    Does preprocessing to the image and cuts outs tables. Then save image and all cut out rois
    as different files.

    Args:
        image: path to image
        tables: list of extracted table annotations
        target: folder to save the results in
        file_name: name of image
        text: list of extracted text region annotations
    """
    # create function to convert PIL Image to torch Tensor
    to_tensor = transforms.PILToTensor()

    # create new folder for image files
    target = f"{target}/{file_name}/"
    os.makedirs(target, exist_ok=True)

    # preprocessing image
    img = Image.open(image)
    img = ImageOps.exif_transpose(img)  # turns image in right orientation

    # save image as jpg and pt
    img.save(f"{target}/" + file_name + ".jpg")
    torch.save(to_tensor(img), f"{target}/" + file_name + ".pt")

    # save one file for every roi
    tablelist = []
    for idx, tab in enumerate(tables):
        # get table coords
        coord = tab["coords"]
        tablelist.append(coord)

        # crop table from image
        tableimg = img.crop((coord))  # type: ignore

        # save image of table
        torch.save(
            to_tensor(tableimg), f"{target}/" + file_name + "_table_" + str(idx) + ".pt"
        )
        tableimg.save(f"{target}/" + file_name + "_table_" + str(idx) + ".jpg")

        # cell bounding boxs (naming: image_file_name _ cell _ idx . pt)
        cells = torch.tensor(tab["cells"])
        torch.save(cells, f"{target}/" + file_name + "_cell_" + str(idx) + ".pt")

        # column bounding boxs (naming: image_file_name _ col _ idx . pt)
        columns = torch.tensor(tab["columns"])
        torch.save(columns, f"{target}/" + file_name + "_col_" + str(idx) + ".pt")

        # row bounding boxs (naming: image_file_name _ row _ idx . pt)
        rows = torch.tensor(tab["rows"])
        torch.save(rows, f"{target}/" + file_name + "_row_" + str(idx) + ".pt")

    # save table bounding boxs (naming: image_file_name _ tables . pt)
    if tablelist:
        table = torch.tensor(tablelist)
        torch.save(table, f"{target}/" + file_name + "_tables.pt")

    # save text bounding boxes (naming: image_file_name _ texts . pt)
    textlist = []
    if text:
        for region in text:
            textlist.append(region["coords"])

        texts = torch.tensor(textlist)
        torch.save(texts, f"{target}/" + file_name + "_textregions" + ".pt")


def main(
    datafolder: str, imgfolder: str, targetfolder: str, ignore_empty: bool = True
) -> None:
    """Main function for preprocessing the datasets.

    Takes the folder of a dataset and preprocesses it, then saves preprocessed images and files
    with bounding boxes.

    table.pt: file with bounding boxes of tables
    text.pt: file with bounding boxes of text
    cell.pt: file with bounding boxes of cell
    column.pt: file with bounding boxes of column
    row.pt: file with bounding boxes of row

    format is always (N x (top_left_x, top_left_y, bottom_right_x, bottom_right_y))

    Args:
        datafolder: path to folder containing raw annotations
        imgfolder: path to folder containing raw images
        targetfolder: folder to save preprocessed dataset
        ignore_empty: if true images with no annotated tables are ignored

    """
    print("Processing folder, this may take a little while!")

    files = [x for x in glob.glob(f"{datafolder}/*.xml")]

    file_names = [os.path.splitext(os.path.basename(path))[0] for path in files]
    images = [f"{imgfolder}/{x}.jpg" for x in file_names]

    for file_name, file, img in tqdm(
        zip(file_names, files, images), desc="preprocessing", total=len(files)
    ):

        # check for strange files
        if plt.imread(img).ndim == 3:
            table, text = extract_annotation(file)
            if table or not ignore_empty:
                preprocess(img, table, targetfolder, file_name, text)


def get_args() -> argparse.Namespace:
    """Defines arguments.

    Returns:
        parsed arguments.
    """
    parser = argparse.ArgumentParser(description="preprocess")
    parser.add_argument("--BonnData", action=argparse.BooleanOptionalAction)
    parser.set_defaults(BonnData=False)

    parser.add_argument("--GloSAT", action=argparse.BooleanOptionalAction)
    parser.set_defaults(GloSAT=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    ours = args.BonnData
    glosat = args.GloSAT

    if not ours and not glosat:
        print("No data to preprocess given.")

    if ours:
        datafolder = (
            f"{Path(__file__).parent.absolute()}/../../data/BonnData/annotations"
        )
        imgfolder = f"{Path(__file__).parent.absolute()}/../../data/BonnData/images"
        targetfolder = (
            f"{Path(__file__).parent.absolute()}/../../data/BonnData/preprocessed"
        )
        main(
            datafolder=datafolder,
            imgfolder=imgfolder,
            targetfolder=targetfolder,
        )

        print(f"datafolder {datafolder}")
        print(f"fimgfolder: {imgfolder}")
        print(f"targetfolder: {targetfolder}")

    if glosat:
        main(
            datafolder=f"{Path(__file__).parent.absolute()}/../../data/"
            f"GloSAT/datasets/Train/Fine/Transkribus",
            imgfolder=f"{Path(__file__).parent.absolute()}/../../data/"
            f"GloSAT/datasets/Train/JPEGImages",
            targetfolder=f"{Path(__file__).parent.absolute()}/../../data/"
            f"GloSAT/preprocessed",
        )

        main(
            datafolder=f"{Path(__file__).parent.absolute()}/../../data/"
            f"GloSAT/datasets/Test/Fine/Transkribus",
            imgfolder=f"{Path(__file__).parent.absolute()}/../../data/"
            f"GloSAT/datasets/Test/JPEGImages",
            targetfolder=f"{Path(__file__).parent.absolute()}/../../data/"
            f"GloSAT/preprocessed",
        )
