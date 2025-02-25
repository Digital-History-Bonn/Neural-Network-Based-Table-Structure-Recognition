"""Postprocessing."""
import argparse
import glob
import json
import os
from pathlib import Path
from typing import Literal, Tuple

import pandas
import torch
from lightning.fabric.utilities import move_data_to_device
from torchvision.io import read_image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)

from tqdm import tqdm
from transformers import AutoModelForObjectDetection
from typing_extensions import List


from src.historicdocumentprocessing.util.metricsutil import calcstats_iodt, calcstats_overlap, calcmetric_overlap, \
    calcstats_iou, calcmetric, get_dataframe
from src.historicdocumentprocessing.tabletransformer_dataset import CustomDataset
from src.historicdocumentprocessing.util.glosat_paper_postprocessing_method import (
    reconstruct_table,
)
from src.historicdocumentprocessing.util.tablesutil import (
    clustertablesseperately,
    getcells,
    getsurroundingtable,
    remove_invalid_bbox, reversetablerelativebboxes_outer, boxoverlap, extractboxes,
)


def postprocess_rcnn_sep(
    modelpath=None,
    targetloc=None,
    datasetname=None,
    filter: bool = False,
    iou_thresholds: List[float] = None,  # [0.5, 0.6, 0.7, 0.8, 0.9]
    epsprefactor=tuple([3.0, 1.5]),
    minsamples: List[int] = None,  # [2, 3]
):
    """Seperately calculate metrics for table coords by DBSCAN clustering image cells into tables and for GloSAT Cell Postprocessing using ground truth tables.

    Args:
        modelpath: model path
        targetloc: ground truth file location
        datasetname: name of dataset
        filter: wether to filter results
        iou_thresholds: iou threshold
        epsprefactor: prefactor for DBSCAN parameter eps
        minsamples: DBSCAN parameter minsamples

    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    if minsamples is None:
        minsamples = [2, 3]
    saveloc = f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/testevalfinal1/fullimg/{datasetname}"
    tableioulist = []
    tablef1list = []
    tablewf1list = []
    tabletpsum = torch.zeros(len(iou_thresholds))
    tablefpsum = torch.zeros(len(iou_thresholds))
    tablefnsum = torch.zeros(len(iou_thresholds))
    tableimagedf = pandas.DataFrame(
        columns=["img", "mean pred iou", "mean tar iou", "wf1", "prednum"]
    )
    # --------------------------------------------------------------------------
    cellioulist = []
    cellf1list = []
    cellwf1list = []
    celltpsum = torch.zeros(len(iou_thresholds))
    cellfpsum = torch.zeros(len(iou_thresholds))
    cellfnsum = torch.zeros(len(iou_thresholds))
    cellimagedf = pandas.DataFrame(
        columns=["img", "mean pred iou", "mean tar iou", "wf1", "prednum"]
    )
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        **{"box_detections_per_img": 200},
    )
    model.load_state_dict(torch.load(modelpath))
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        model.eval()
    else:
        print("Cuda not available")
        return
    modelname = modelpath.split(os.sep)[-1]
    saveloc = f"{saveloc}/{modelname}/iou_{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}/postprocessed"
    if filter:
        with open(
            f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/bestfilterthresholds/{modelname}.txt",
            "r",
        ) as f:
            filtering = float(f.read())
        saveloc = f"{saveloc}_filtering_{filtering}"
    saveloc = f"{saveloc}/eps_{epsprefactor[0]}_{epsprefactor[1]}/minsamples_{minsamples[0]}_{minsamples[1]}/seperateeval"
    for n, folder in tqdm(enumerate(glob.glob(f"{targetloc}/*"))):
        # print(folder)
        impath = f"{folder}/{folder.split('/')[-1]}.jpg"
        imname = folder.split("/")[-1]
        img = (read_image(impath) / 255).to(device)
        output = model([img])
        output = {k: v.detach().cpu() for k, v in output[0].items()}
        # print(output['boxes'], output['boxes'][output['scores']>0.8])
        if filter:
            output["boxes"] = output["boxes"][output["scores"] > filtering]
        fullimagepredbox = remove_invalid_bbox(output["boxes"])
        tableprediou, tabletargetiou, tabletp, tablefp, tablefn = (
            postprocess_eval_tablesep(
                fullimagepredbox=fullimagepredbox,
                imname=imname,
                targetloc=targetloc,
                minsamples=minsamples,
                epsprefactorchange=epsprefactor,
            )
        )
        tableprec, tablerec, tablef1, tablewf1 = calcmetric(
            tp=tabletp, fp=tablefp, fn=tablefn, iou_thresholds=iou_thresholds
        )
        tabletpsum += tabletp
        tablefpsum += tablefp
        tablefnsum += tablefn
        tableioulist.append(tableprediou)
        tablef1list.append(tablef1)
        tablewf1list.append(tablewf1)

        tableimagemetrics = {
            "img": imname,
            "mean pred iou": torch.mean(tableprediou).item(),
            "mean tar iou": torch.mean(tabletargetiou).item(),
            "wf1": tablewf1.item(),
            "prednum": fullimagepredbox.shape[0],
        }
        tableimagemetrics.update(
            {
                f"prec@{iou_thresholds[i]}": tableprec[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        tableimagemetrics.update(
            {
                f"recall@{iou_thresholds[i]}": tablerec[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        tableimagemetrics.update(
            {
                f"f1@{iou_thresholds[i]}": tablef1[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        tableimagemetrics.update(
            {
                f"tp@{iou_thresholds[i]}": tabletp[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        tableimagemetrics.update(
            {
                f"fp@{iou_thresholds[i]}": tablefp[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        tableimagemetrics.update(
            {
                f"fn@{iou_thresholds[i]}": tablefn[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        tableimagedf = pandas.concat(
            [tableimagedf, pandas.DataFrame(tableimagemetrics, index=[n])]
        )

        # -----------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------

        cellprediou, celltargetiou, celltp, cellfp, cellfn = postprocess_eval_cellsep(
            fullimagepredbox=fullimagepredbox,
            imname=imname,
            targetloc=targetloc,
            iou_thresholds=iou_thresholds,
        )
        # print(calcstats(fullimagepredbox, fullimagegroundbox,
        #                                                               iou_thresholds=iou_thresholds, imname=preds.split('/')[-1]), fullfp, targets[0])
        cellprec, cellrec, cellf1, cellwf1 = calcmetric(
            celltp, cellfp, cellfn, iou_thresholds=iou_thresholds
        )
        celltpsum += celltp
        cellfpsum += cellfp
        cellfnsum += cellfn
        cellioulist.append(cellprediou)
        cellf1list.append(cellf1)
        cellwf1list.append(cellwf1)

        cellimagemetrics = {
            "img": imname,
            "mean pred iou": torch.mean(cellprediou).item(),
            "mean tar iou": torch.mean(celltargetiou).item(),
            "wf1": cellwf1.item(),
            "prednum": fullimagepredbox.shape[0],
        }
        cellimagemetrics.update(
            {
                f"prec@{iou_thresholds[i]}": cellprec[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        cellimagemetrics.update(
            {
                f"recall@{iou_thresholds[i]}": cellrec[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        cellimagemetrics.update(
            {
                f"f1@{iou_thresholds[i]}": cellf1[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        cellimagemetrics.update(
            {
                f"tp@{iou_thresholds[i]}": celltp[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        cellimagemetrics.update(
            {
                f"fp@{iou_thresholds[i]}": cellfp[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        cellimagemetrics.update(
            {
                f"fn@{iou_thresholds[i]}": cellfn[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        cellimagedf = pandas.concat(
            [cellimagedf, pandas.DataFrame(cellimagemetrics, index=[n])]
        )

    totaltabledf = get_dataframe(fnsum=tablefnsum, fpsum=tablefpsum, tpsum=tabletpsum)
    totalcelldf = get_dataframe(fnsum=cellfnsum, tpsum=celltpsum, fpsum=cellfpsum)
    conclusiondf = pandas.DataFrame(columns=["wf1"])
    conclusiondf = pandas.concat(
        [
            conclusiondf,
            pandas.DataFrame(totaltabledf, index=["table postprocessing iou"]),
            pandas.DataFrame(
                totalcelldf, index=["cell postprocessing iou (tables given)"]
            ),
        ]
    )
    os.makedirs(saveloc, exist_ok=True)
    conclusiondf.to_csv(f"{saveloc}/conclusiondf.csv")
    tableimagedf.to_csv(f"{saveloc}/tableiou.csv")
    cellimagedf.to_csv(f"{saveloc}/celliou.csv")


def postprocess_kosmos_sep(
    targetloc: str = None,  # f"{Path(__file__).parent.absolute()}/../../data/BonnData/test"
    predloc: str = None,  # f"{Path(__file__).parent.absolute()}/../../results/kosmos25/BonnData/Tabellen/test"
    datasetname: str = "BonnData",
    iou_thresholds: List[float] = None,  # [0.5, 0.6, 0.7, 0.8, 0.9]
    epsprefactorchange=tuple([3.0, 1.5]),
    minsamples: List[int] = None,  # [2, 3]  # [4, 5]
):
    """Seperately calculate metrics for table coords by DBSCAN clustering image cells into tables and for GloSAT Cell Postprocessing using ground truth tables.

    Args:
        targetloc: ground truth file location
        predloc: location of predictions
        datasetname: name of dataset
        iou_thresholds: iou threshold
        epsprefactorchange: prefactor for DBSCAN parameter eps
        minsamples: DBSCAN parameter minsamples

    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    if minsamples is None:
        minsamples = [2, 3]
    saveloc = f"{Path(__file__).parent.absolute()}/../../results/kosmos25/testevalfinal1/fullimg/{datasetname}"
    saveloc = f"{saveloc}/iou_{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}/postprocessed/eps_{epsprefactorchange[0]}_{epsprefactorchange[1]}/minsamples_{minsamples[0]}_{minsamples[1]}/seperateeval"
    tableioulist = []
    tablef1list = []
    tablewf1list = []
    tabletpsum = torch.zeros(len(iou_thresholds))
    tablefpsum = torch.zeros(len(iou_thresholds))
    tablefnsum = torch.zeros(len(iou_thresholds))
    tableimagedf = pandas.DataFrame(
        columns=["img", "mean pred iou", "mean tar iou", "wf1", "prednum"]
    )
    # --------------------------------------------------------------------------
    cellioulist = []
    cellf1list = []
    cellwf1list = []
    celltpsum = torch.zeros(len(iou_thresholds))
    cellfpsum = torch.zeros(len(iou_thresholds))
    cellfnsum = torch.zeros(len(iou_thresholds))
    cellimagedf = pandas.DataFrame(
        columns=["img", "mean pred iou", "mean tar iou", "wf1", "prednum"]
    )

    for n, targets in tqdm(enumerate(glob.glob(f"{targetloc}/*"))):
        preds = glob.glob(f"{predloc}/{targets.split('/')[-1]}")[0]
        # imagepred = [file for file in glob.glob(f"{preds}/*.json") if "_table_" not in file][0]
        # bboxes on full image
        # fullimagepred = glob.glob(f"{preds}/(?!.*_table_)^.*$")[0]
        # print(preds, targets)
        fullimagepred = [
            file for file in glob.glob(f"{preds}/*") if "_table_" not in file
        ][0]
        # print(fullimagepred)
        imname = preds.split("/")[-1]
        with open(fullimagepred) as p:
            fullimagepredbox = remove_invalid_bbox(
                extractboxes(json.load(p), fpath=None)
            )

        tableprediou, tabletargetiou, tabletp, tablefp, tablefn = (
            postprocess_eval_tablesep(
                fullimagepredbox=fullimagepredbox,
                imname=imname,
                targetloc=targetloc,
                minsamples=minsamples,
                epsprefactorchange=epsprefactorchange,
            )
        )
        tableprec, tablerec, tablef1, tablewf1 = calcmetric(
            tp=tabletp, fp=tablefp, fn=tablefn, iou_thresholds=iou_thresholds
        )
        tabletpsum += tabletp
        tablefpsum += tablefp
        tablefnsum += tablefn
        tableioulist.append(tableprediou)
        tablef1list.append(tablef1)
        tablewf1list.append(tablewf1)

        tableimagemetrics = {
            "img": imname,
            "mean pred iou": torch.mean(tableprediou).item(),
            "mean tar iou": torch.mean(tabletargetiou).item(),
            "wf1": tablewf1.item(),
            "prednum": fullimagepredbox.shape[0],
        }
        tableimagemetrics.update(
            {
                f"prec@{iou_thresholds[i]}": tableprec[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        tableimagemetrics.update(
            {
                f"recall@{iou_thresholds[i]}": tablerec[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        tableimagemetrics.update(
            {
                f"f1@{iou_thresholds[i]}": tablef1[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        tableimagemetrics.update(
            {
                f"tp@{iou_thresholds[i]}": tabletp[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        tableimagemetrics.update(
            {
                f"fp@{iou_thresholds[i]}": tablefp[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        tableimagemetrics.update(
            {
                f"fn@{iou_thresholds[i]}": tablefn[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        tableimagedf = pandas.concat(
            [tableimagedf, pandas.DataFrame(tableimagemetrics, index=[n])]
        )

        # -----------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------

        cellprediou, celltargetiou, celltp, cellfp, cellfn = postprocess_eval_cellsep(
            fullimagepredbox=fullimagepredbox,
            imname=imname,
            targetloc=targetloc,
            iou_thresholds=iou_thresholds,
        )
        # print(calcstats(fullimagepredbox, fullimagegroundbox,
        #                                                               iou_thresholds=iou_thresholds, imname=preds.split('/')[-1]), fullfp, targets[0])
        cellprec, cellrec, cellf1, cellwf1 = calcmetric(
            celltp, cellfp, cellfn, iou_thresholds=iou_thresholds
        )
        celltpsum += celltp
        cellfpsum += cellfp
        cellfnsum += cellfn
        cellioulist.append(cellprediou)
        cellf1list.append(cellf1)
        cellwf1list.append(cellwf1)

        cellimagemetrics = {
            "img": imname,
            "mean pred iou": torch.mean(cellprediou).item(),
            "mean tar iou": torch.mean(celltargetiou).item(),
            "wf1": cellwf1.item(),
            "prednum": fullimagepredbox.shape[0],
        }
        cellimagemetrics.update(
            {
                f"prec@{iou_thresholds[i]}": cellprec[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        cellimagemetrics.update(
            {
                f"recall@{iou_thresholds[i]}": cellrec[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        cellimagemetrics.update(
            {
                f"f1@{iou_thresholds[i]}": cellf1[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        cellimagemetrics.update(
            {
                f"tp@{iou_thresholds[i]}": celltp[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        cellimagemetrics.update(
            {
                f"fp@{iou_thresholds[i]}": cellfp[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        cellimagemetrics.update(
            {
                f"fn@{iou_thresholds[i]}": cellfn[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        cellimagedf = pandas.concat(
            [cellimagedf, pandas.DataFrame(cellimagemetrics, index=[n])]
        )

    totaltabledf = get_dataframe(fnsum=tablefnsum, fpsum=tablefpsum, tpsum=tabletpsum)
    totalcelldf = get_dataframe(fnsum=cellfnsum, tpsum=celltpsum, fpsum=cellfpsum)
    conclusiondf = pandas.DataFrame(columns=["wf1"])
    conclusiondf = pandas.concat(
        [
            conclusiondf,
            pandas.DataFrame(totaltabledf, index=["table postprocessing iou"]),
            pandas.DataFrame(
                totalcelldf, index=["cell postprocessing iou (tables given)"]
            ),
        ]
    )
    os.makedirs(saveloc, exist_ok=True)
    conclusiondf.to_csv(f"{saveloc}/conclusiondf.csv")
    tableimagedf.to_csv(f"{saveloc}/tableiou.csv")
    cellimagedf.to_csv(f"{saveloc}/celliou.csv")


def postprocess_eval_tablesep(
    fullimagepredbox: torch.Tensor,
    imname: str,
    includeoutlier: bool = False,
    epsprefactorchange=None,
    minsamples: List[int] = None,  # [2, 3]  # [4, 5]
    targetloc: str = None,
    iou_thresholds: List[float] = None,  # [0.5, 0.6, 0.7, 0.8, 0.9]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluation for table coords extracted by DBSCAN clustering image cells into tables.

    Args:
        fullimagepredbox: cell predictions
        imname: image name
        includeoutlier: wether to include outliers while clustering
        iou_thresholds: iou threshold
        epsprefactorchange: prefactor for DBSCAN parameter eps
        minsamples: DBSCAN parameter minsamples
        targetloc: ground truth file location

    Returns:
        iou stats
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    if minsamples is None:
        minsamples = [2, 3]
    epsprefactor = tuple([3.0, 1.5]) if not epsprefactorchange else epsprefactorchange
    clusteredtables = clustertablesseperately(
        fullimagepredbox,
        epsprefactor=epsprefactor,
        includeoutlier=includeoutlier,
        minsamples=minsamples,
    )
    tablecoords = []
    for _i, t in enumerate(clusteredtables):
        tablecoord = getsurroundingtable(t).tolist()
        tablecoords.append(tablecoord)
    tablecoords = torch.tensor(tablecoords)
    tablepath = f"{targetloc}/{imname}/{imname}_tables.pt"
    tablebboxes = torch.load(tablepath)
    return calcstats_iou(
        predbox=tablecoords,
        targetbox=tablebboxes,
        iou_thresholds=iou_thresholds,
        imname=imname,
    )


def postprocess_eval_cellsep(
    fullimagepredbox: torch.Tensor,
    imname: str,
    targetloc: str = None,
    iou_thresholds: List[float] = None,  # [0.5, 0.6, 0.7, 0.8, 0.9]
    tablerelative: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate GloSAT Cell Postprocessing using ground truth tables.

    Args:
        fullimagepredbox: image predictions
        imname: image names
        targetloc: target folder
        iou_thresholds: iou thresholds
        tablerelative: wether groundtruth coordinates are table relative

    Returns:
        iou stats
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    tablepath = f"{targetloc}/{imname}/{imname}_tables.pt"
    tablebboxes = torch.load(tablepath)
    finalcells = []
    for table in tablebboxes:
        tablepreds = []
        for preds in fullimagepredbox:
            if boxoverlap(preds, table, fuzzy=0):
                tablepreds.append(preds.tolist())
        if tablepreds:
            rows, colls = reconstruct_table(
                cells=tablepreds, table=table.tolist(), eps=4
            )
            cells = reconstruct_bboxes(rows, colls, table.tolist())
            finalcells += cells.tolist()
    finalcells = (
        torch.tensor(finalcells).squeeze() if finalcells else torch.empty((0, 4))
    )
    if finalcells.dim() == 1:
        finalcells = torch.unsqueeze(finalcells, 0)
    groundfolder = f"{targetloc}/{imname}"
    if tablerelative:
        fullimagegroundbox = reversetablerelativebboxes_outer(groundfolder)
    else:
        fullimagegroundbox = torch.load(
            glob.glob(f"{groundfolder}/{groundfolder.split('/')[-1]}.pt")[0]
        )
    return calcstats_iou(
        predbox=finalcells,
        targetbox=fullimagegroundbox,
        iou_thresholds=iou_thresholds,
        imname=imname,
    )


def reconstruct_bboxes(rows: list, colls: list, tablecoords: list) -> torch.Tensor:
    """Reconstruct BBoxes from output of GloSAT postprocessing.

    Args:
        rows: rows
        colls: columns
        tablecoords: table coordinates

    Returns:
        reconstructed BBoxes

    """
    rows = rows + [tablecoords[1], tablecoords[3]]
    colls = colls + [tablecoords[0], tablecoords[2]]
    rows = list(set(rows))
    colls = list(set(colls))
    rows.sort()
    colls.sort()
    boxes = []
    assert len(rows) >= 2 and len(colls) >= 2
    for i in range(len(rows) - 1):
        for j in range(len(colls) - 1):
            newbox = [colls[j], rows[i], colls[j + 1], rows[i + 1]]
            boxes.append(newbox)
    return torch.tensor(boxes)


def postprocess_kosmos(
    targetloc: str,  # = f"{Path(__file__).parent.absolute()}/../../data/BonnData/test"
    predloc: str,  # = f"{Path(__file__).parent.absolute()}/../../results/kosmos25/BonnData/Tabellen/test"
    datasetname: str,  # = "BonnData"
):
    """Postprocess kosmos.

    Args:
        targetloc: target folder
        predloc: prediction folder
        datasetname: name of dataset

    """
    saveloc = f"{Path(__file__).parent.absolute()}/../../results/kosmos25/postprocessed/fullimg/{datasetname}"
    for _n, targets in tqdm(enumerate(glob.glob(f"{targetloc}/*"))):
        preds = glob.glob(f"{predloc}/{targets.split('/')[-1]}")[0]
        # bboxes on full image
        fullimagepred = [
            file for file in glob.glob(f"{preds}/*") if "_table_" not in file
        ][0]
        imname = preds.split("/")[-1]
        with open(fullimagepred) as p:
            fullimagepredbox = remove_invalid_bbox(
                extractboxes(json.load(p), fpath=None)
            )
            postprocess(fullimagepredbox, imname, saveloc)


def postprocess(
    fullimagepredbox: torch.Tensor,
    imname: str,
    saveloc: str,
    includeoutlier: bool = False,
    includeoutliers_cellsearch: bool = False,
    saveempty: bool = False,
    epsprefactorchange=None,
    minsamples: List[int] = None,  # [2, 3],  # [4, 5]
) -> torch.Tensor:
    """Inner method for postprocessing, cluster tables and apply GloSAT postprocessing.

    Args:
        fullimagepredbox: prediction
        imname: image name
        saveloc: location to save to
        includeoutlier: wether to include outliers while clustering tables
        includeoutliers_cellsearch: wether to include outliers in GloSAT postprocessing
        saveempty: wether to save empty result files
        epsprefactorchange: prefactor for DBSCAN parameter eps
        minsamples: DBSCAN parameter minsamples

    Returns:
        extracted BBoxes

    """
    if minsamples is None:
        minsamples = [2, 3]
    epsprefactor = tuple([3.0, 1.5]) if not epsprefactorchange else epsprefactorchange
    clusteredtables = clustertablesseperately(
        fullimagepredbox,
        epsprefactor=epsprefactor,
        includeoutlier=includeoutlier,
        minsamples=minsamples,
    )
    boxes = []
    saveloc = f"{saveloc}/eps_{epsprefactor[0]}_{epsprefactor[1]}/minsamples_{minsamples[0]}_{minsamples[1]}"
    if not includeoutlier:
        saveloc = f"{saveloc}/withoutoutlier"
    else:
        saveloc = f"{saveloc}/withoutlier"
    saveloc = f"{f'{saveloc}_includeoutliers_glosat' if includeoutliers_cellsearch else f'{saveloc}_excludeoutliers_glosat'}"
    saveloc = f"{saveloc}/{imname}"
    for _i, t in enumerate(clusteredtables):
        tablecoord = getsurroundingtable(t).tolist()
        rows, colls = reconstruct_table(
            t.tolist(), tablecoord, eps=4, include_outliers=includeoutliers_cellsearch
        )
        tbboxes = reconstruct_bboxes(rows, colls, tablecoord)
        boxes.append(tbboxes)
    if len(boxes) > 0:
        bboxes = torch.vstack(boxes)
        os.makedirs(saveloc, exist_ok=True)
        torch.save(bboxes, f"{saveloc}/{imname}.pt")
        return bboxes
    else:
        print(f"no valid predictions for image {imname}")
        if saveempty:
            os.makedirs(saveloc, exist_ok=True)
            torch.save(torch.empty((0, 4)), f"{saveloc}/{imname}.pt")
        return torch.empty((0, 4))


def postprocess_rcnn(
    modelname=None, targetloc=None, datasetname=None, filter: bool = False, valid=True
):
    """Postprocess rcnn.

    Args:
        modelname: modelname
        targetloc: ground truth location
        datasetname: name of dataset
        filter: wether to filer rcnn output
        valid: wether to use valid filter threshold

    """
    saveloc = f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/postprocessed/fullimg/{datasetname}"
    modelpath = f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/{modelname}"
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        **{"box_detections_per_img": 200},
    )
    model.load_state_dict(torch.load(modelpath))
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        model.eval()
    else:
        print("Cuda not available")
        return
    modelname = modelpath.split(os.sep)[-1]
    saveloc = f"{saveloc}/{modelname}"
    if filter:
        with open(
            f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn{'_valid' if valid else ''}/bestfilterthresholds/{modelname}.txt",
            "r",
        ) as f:
            filtering = float(f.read())
        saveloc = f"{saveloc}_filtering_{filtering}{'_valid' if valid else ''}"
    for _n, folder in tqdm(enumerate(glob.glob(f"{targetloc}/*"))):
        impath = f"{folder}/{folder.split('/')[-1]}.jpg"
        imname = folder.split("/")[-1]
        img = (read_image(impath) / 255).to(device)
        output = model([img])
        output = {k: v.detach().cpu() for k, v in output[0].items()}
        if filter:
            output["boxes"] = output["boxes"][output["scores"] > filtering]
        fullimagepredbox = remove_invalid_bbox(output["boxes"])
        postprocess(fullimagepredbox, imname, saveloc)


def postprocess_tabletransformer(
    modelname: str = None,
    targetloc: str = None,
    datasetname: str = None,
    filter: bool = False,
    valid=True,
):
    """Postprocess tabletransformer.

    Args:
        modelname: model name
        targetloc: ground truth folder
        datasetname: name of dataset
        filter: wether to filter results
        valid: wether to use valid filter threshold

    """
    saveloc = f"{Path(__file__).parent.absolute()}/../../results/tabletransformer/postprocessed/fullimg/{datasetname}"
    model = AutoModelForObjectDetection.from_pretrained(
        "microsoft/table-transformer-structure-recognition"
    )
    modelpath = f"{Path(__file__).parent.absolute()}/../../checkpoints/tabletransformer/{modelname}"
    if "aachen" in modelname:
        model.load_state_dict(torch.load(modelpath, map_location="cuda:0"))
    else:
        model.load_state_dict(torch.load(modelpath))
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        model.eval()
    else:
        print("Cuda not available")
        return
    saveloc = f"{saveloc}/{modelname}"
    if filter:
        with open(
            f"{Path(__file__).parent.absolute()}/../../results/tabletransformer/bestfilterthresholds{'_valid' if valid else ''}/{modelname}.txt",
            "r",
        ) as f:
            filtering = float(f.read())
        saveloc = f"{saveloc}_filtering_{filtering}{'_valid' if valid else ''}"
    dataset = CustomDataset(
        targetloc,
        "fullimage",
        transforms=None,
    )
    for i in tqdm(range(len(dataset))):
        img, _, _ = dataset.getimgtarget(i, addtables=False)
        encoding = move_data_to_device(
            dataset.ImageProcessor(img, return_tensors="pt"), device=device
        )
        folder = dataset.getfolder(i)
        imname = folder.split("/")[-1]
        if not os.path.isfile(
            f"{saveloc}/eps_3.0_1.5/minsamples_2_3/withoutoutlier_excludeoutliers_glosat/{imname}/{imname}.pt"
        ):
            with torch.no_grad():
                results = model(**encoding)
            width, height = img.size
            if filter:
                output = dataset.ImageProcessor.post_process_object_detection(
                    results, threshold=filtering, target_sizes=[(height, width)]
                )[0]
            else:
                output = dataset.ImageProcessor.post_process_object_detection(
                    results, threshold=0.0, target_sizes=[(height, width)]
                )[0]
            boxes = getcells(
                rows=output["boxes"][output["labels"] == 2],
                cols=output["boxes"][output["labels"] == 1],
            )  # must be cells for postprocessing algo to work
            fullimagepredbox = remove_invalid_bbox(boxes).to("cpu")
            postprocess(
                fullimagepredbox=fullimagepredbox, imname=imname, saveloc=saveloc
            )


def postprocess_eval(
    datasetname: str,
    modeltype: Literal["kosmos25", "fasterrcnn", "tabletransformer"] = "kosmos25",
    targetloc: str = None,
    modelname: str = None,
    tablerelative=True,
    iou_thresholds: List[float] = None,  # [0.5, 0.6, 0.7, 0.8, 0.9]
    # tableareaonly=False,
    includeoutliers_cellsearch=False,
    filter: bool = False,
    valid: bool = True,
    epsprefactor=tuple([3.0, 1.5]),
    minsamples: List[int] = None,  # [2, 3],  # [4, 5]
):
    """Evaluate postprocessing results.

    Args:
        datasetname: datasetname
        modeltype: type of model
        targetloc: ground truth folder
        modelname: modelname (for fasterrcnn and tabletransformer)
        tablerelative: wether ground truth is table relative
        iou_thresholds: iou thresholds
        includeoutliers_cellsearch: wether to include outliers in GloSAT postprocessing
        filter: wether to filter results
        valid: wether to use valid filter threshold
        epsprefactor: prefactor for DBSCAN parameter eps
        minsamples: DBSCAN parameter minsamples

    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    if minsamples is None:
        minsamples = [2, 3]
    predloc = f"{Path(__file__).parent.absolute()}/../../results/{modeltype}/postprocessed/fullimg/{datasetname}"
    saveloc = f"{Path(__file__).parent.absolute()}/../../results/{modeltype}/testevalfinal1/fullimg/{datasetname}"
    if modelname:
        predloc = f"{predloc}/{modelname}"
        if filter:
            with open(
                f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/bestfilterthresholds/{modelname}.txt",
                "r",
            ) as f:
                filtering = float(f.read())
            predloc = f"{predloc}_filtering_{filtering}"
            if valid:
                predloc = f"{predloc}_valid"
        saveloc = f"{saveloc}/{modelname}/iou_{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}/postprocessed/eps_{epsprefactor[0]}_{epsprefactor[1]}/minsamples_{minsamples[0]}_{minsamples[1]}/withoutoutlier"
    else:
        saveloc = f"{saveloc}/iou_{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}/postprocessed/eps_{epsprefactor[0]}_{epsprefactor[1]}/minsamples_{minsamples[0]}_{minsamples[1]}/withoutoutlier"
    saveloc = f"{f'{saveloc}/includeoutliers_glosat' if includeoutliers_cellsearch else f'{saveloc}/excludeoutliers_glosat'}"
    if filter and modelname:
        saveloc = f"{saveloc}_filtering_optimal_{filtering}"
    predloc = f"{predloc}/eps_{epsprefactor[0]}_{epsprefactor[1]}/minsamples_{minsamples[0]}_{minsamples[1]}/withoutoutlier"
    predloc = f"{f'{predloc}' if includeoutliers_cellsearch else f'{predloc}_excludeoutliers_glosat'}"
    # saveloc = f"{Path(__file__).parent.absolute()}/../../results/{modeltype}/testevalfinal1/fullimg/{datasetname}/{modelname}/iou_{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}/postprocessed"
    # ## initializing variables ## #
    fullioulist = []
    fullf1list = []
    fullwf1list = []
    fulltpsum = torch.zeros(len(iou_thresholds))
    fullfpsum = torch.zeros(len(iou_thresholds))
    fullfnsum = torch.zeros(len(iou_thresholds))
    fulltpsum_predonly = torch.zeros(len(iou_thresholds))
    fullfpsum_predonly = torch.zeros(len(iou_thresholds))
    fullfnsum_predonly = torch.zeros(len(iou_thresholds))
    tpsum_overlap = torch.zeros(1)
    fpsum_overlap = torch.zeros(1)
    fnsum_overlap = torch.zeros(1)
    tpsum_overlap_predonly = torch.zeros(1)
    fpsum_overlap_predonly = torch.zeros(1)
    fnsum_overlap_predonly = torch.zeros(1)
    tpsum_iodt = torch.zeros(len(iou_thresholds))
    fpsum_iodt = torch.zeros(len(iou_thresholds))
    fnsum_iodt = torch.zeros(len(iou_thresholds))
    tpsum_iodt_predonly = torch.zeros(len(iou_thresholds))
    fpsum_iodt_predonly = torch.zeros(len(iou_thresholds))
    fnsum_iodt_predonly = torch.zeros(len(iou_thresholds))
    predcount = 0
    overlapdf = pandas.DataFrame(columns=["img", "prednum"])
    fullimagedf = pandas.DataFrame(
        columns=["img", "mean pred iou", "mean tar iou", "wf1", "prednum"]
    )
    iodtdf = pandas.DataFrame(
        columns=["img", "mean pred iod", "mean tar iod", "wf1", "prednum"]
    )
    # ## initializing variables ## #
    for n, pred in tqdm(enumerate(glob.glob(f"{predloc}/*"))):
        folder = f"{targetloc}/{pred.split('/')[-1]}"
        imname = folder.split("/")[-1]
        fullimagepredbox = torch.load(glob.glob(f"{pred}/{pred.split('/')[-1]}.pt")[0])
        print(fullimagepredbox)
        # print(folder)
        if tablerelative:
            fullimagegroundbox = reversetablerelativebboxes_outer(folder)
        else:
            fullimagegroundbox = torch.load(
                glob.glob(f"{folder}/{folder.split('/')[-1]}.pt")[0]
            )

        # .......................................
        # fullimagemetrics with IoDT
        # .......................................
        prediod, tariod, iodt_tp, iodt_fp, iodt_fn = calcstats_iodt(
            predbox=fullimagepredbox,
            targetbox=fullimagegroundbox,
            imname=imname,
            iou_thresholds=iou_thresholds,
        )
        iodt_prec, iodt_rec, iodt_f1, iodt_wf1 = calcmetric(
            iodt_tp, iodt_fp, iodt_fn, iou_thresholds=iou_thresholds
        )
        tpsum_iodt += iodt_tp
        fpsum_iodt += iodt_fp
        fnsum_iodt += iodt_fn
        iodtmetrics = {
            "img": imname,
            "mean pred iod": torch.mean(prediod).item(),
            "mean tar iod": torch.mean(tariod).item(),
            "wf1": iodt_wf1.item(),
            "prednum": fullimagepredbox.shape[0],
        }
        iodtmetrics.update(
            {
                f"prec@{iou_thresholds[i]}": iodt_prec[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        iodtmetrics.update(
            {
                f"recall@{iou_thresholds[i]}": iodt_rec[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        iodtmetrics.update(
            {
                f"f1@{iou_thresholds[i]}": iodt_f1[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        iodtmetrics.update(
            {
                f"tp@{iou_thresholds[i]}": iodt_tp[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        iodtmetrics.update(
            {
                f"fp@{iou_thresholds[i]}": iodt_fp[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        iodtmetrics.update(
            {
                f"fn@{iou_thresholds[i]}": iodt_fn[i].item()
                for i in range(len(iou_thresholds))
            }
        )

        iodtdf = pandas.concat([iodtdf, pandas.DataFrame(iodtmetrics, index=[n])])

        # .................................
        # fullimagemetrics with iou
        # .................................
        fullprediou, fulltargetiou, fulltp, fullfp, fullfn = calcstats_iou(
            fullimagepredbox,
            fullimagegroundbox,
            iou_thresholds=iou_thresholds,
            imname=imname,
        )
        # print(calcstats(fullimagepredbox, fullimagegroundbox,
        #                                                               iou_thresholds=iou_thresholds, imname=preds.split('/')[-1]), fullfp, targets[0])
        fullprec, fullrec, fullf1, fullwf1 = calcmetric(
            fulltp, fullfp, fullfn, iou_thresholds=iou_thresholds
        )
        fulltpsum += fulltp
        fullfpsum += fullfp
        fullfnsum += fullfn
        fullioulist.append(fullprediou)
        fullf1list.append(fullf1)
        fullwf1list.append(fullwf1)

        fullimagemetrics = {
            "img": imname,
            "mean pred iou": torch.mean(fullprediou).item(),
            "mean tar iou": torch.mean(fulltargetiou).item(),
            "wf1": fullwf1.item(),
            "prednum": fullimagepredbox.shape[0],
        }
        fullimagemetrics.update(
            {
                f"prec@{iou_thresholds[i]}": fullprec[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        fullimagemetrics.update(
            {
                f"recall@{iou_thresholds[i]}": fullrec[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        fullimagemetrics.update(
            {
                f"f1@{iou_thresholds[i]}": fullf1[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        fullimagemetrics.update(
            {
                f"tp@{iou_thresholds[i]}": fulltp[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        fullimagemetrics.update(
            {
                f"fp@{iou_thresholds[i]}": fullfp[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        fullimagemetrics.update(
            {
                f"fn@{iou_thresholds[i]}": fullfn[i].item()
                for i in range(len(iou_thresholds))
            }
        )

        # ........................................
        # fullimagemetrics with alternate metric
        # ........................................
        fulltp_overlap, fullfp_overlap, fullfn_overlap = calcstats_overlap(
            fullimagepredbox, fullimagegroundbox, imname=imname
        )

        fullprec_overlap, fullrec_overlap, fullf1_overlap = calcmetric_overlap(
            tp=fulltp_overlap, fp=fullfp_overlap, fn=fullfn_overlap
        )
        tpsum_overlap += fulltp_overlap
        fpsum_overlap += fullfp_overlap
        fnsum_overlap += fullfn_overlap
        overlapmetric = {
            "img": imname,
            "prednum": fullimagepredbox.shape[0],
            "prec": fullprec_overlap.item(),
            "recall": fullrec_overlap.item(),
            "f1": fullf1_overlap.item(),
            "tp": fulltp_overlap,
            "fp": fullfp_overlap,
            "fn": fullfn_overlap,
        }

        overlapdf = pandas.concat(
            [overlapdf, pandas.DataFrame(overlapmetric, index=[n])]
        )

        if fullimagepredbox.shape[0] != 0:
            fulltpsum_predonly += fulltp
            fullfpsum_predonly += fullfp
            fullfnsum_predonly += fullfn
            tpsum_overlap_predonly += fulltp_overlap
            fpsum_overlap_predonly += fullfp_overlap
            fnsum_overlap_predonly += fullfn_overlap
            tpsum_iodt_predonly += iodt_tp
            fpsum_iodt_predonly += iodt_fp
            fnsum_iodt_predonly += iodt_fn
            predcount += 1

        fullimagedf = pandas.concat(
            [fullimagedf, pandas.DataFrame(fullimagemetrics, index=[n])]
        )

    totalfullmetrics = get_dataframe(
        fnsum=fullfnsum,
        fpsum=fullfpsum,
        tpsum=fulltpsum,
        nopredcount=fullimagedf.shape[0] - predcount,
        imnum=fullimagedf.shape[0],
        iou_thresholds=iou_thresholds,
    )
    partialfullmetrics = get_dataframe(
        fnsum=fullfnsum_predonly,
        fpsum=fullfpsum_predonly,
        tpsum=fulltpsum_predonly,
        iou_thresholds=iou_thresholds,
    )
    overlapprec, overlaprec, overlapf1 = calcmetric_overlap(
        tp=tpsum_overlap, fp=fpsum_overlap, fn=fnsum_overlap
    )
    overlapprec_predonly, overlaprec_predonly, overlapf1_predonly = calcmetric_overlap(
        tp=tpsum_overlap_predonly, fp=fpsum_overlap_predonly, fn=fnsum_overlap_predonly
    )
    predonlyoverlapdf = pandas.DataFrame(
        {
            "Number of evaluated files": overlapdf.shape[0],
            "Evaluated files without predictions:": overlapdf.shape[0] - predcount,
            "f1": overlapf1_predonly,
            "prec": overlapprec_predonly,
            "recall": overlaprec_predonly,
        },
        index=["overlap with valid preds"],
    )
    totaloverlapdf = pandas.DataFrame(
        {"f1": overlapf1, "prec": overlapprec, "recall": overlaprec}, index=["overlap"]
    )
    totaliodt = get_dataframe(
        fnsum=fnsum_iodt,
        fpsum=fpsum_iodt,
        tpsum=tpsum_iodt,
        nopredcount=iodtdf.shape[0] - predcount,
        imnum=iodtdf.shape[0],
        iou_thresholds=iou_thresholds,
    )
    predonlyiodt = get_dataframe(
        fnsum=fnsum_iodt_predonly,
        fpsum=fpsum_iodt_predonly,
        tpsum=tpsum_iodt_predonly,
        iou_thresholds=iou_thresholds,
    )

    conclusiondf = pandas.DataFrame(columns=["wf1"])

    conclusiondf = pandas.concat(
        [
            conclusiondf,
            pandas.DataFrame(totalfullmetrics, index=["full image IoU"]),
            pandas.DataFrame(
                partialfullmetrics, index=["full image IoU with valid preds"]
            ),
            totaloverlapdf,
            predonlyoverlapdf,
            pandas.DataFrame(totaliodt, index=["full image IoDt"]),
            pandas.DataFrame(predonlyiodt, index=[" full image IoDt with valid preds"]),
        ]
    )
    # save results
    os.makedirs(saveloc, exist_ok=True)
    # print(conclusiondf)
    # print(saveloc)
    overlapdf.to_csv(f"{saveloc}/fullimageoverlapeval.csv")
    fullimagedf.to_csv(f"{saveloc}/fullimageiou.csv")
    iodtdf.to_csv(f"{saveloc}/fullimageiodt.csv")
    conclusiondf.to_csv(f"{saveloc}/overview.csv")


def get_args() -> argparse.Namespace:
    """Define args."""
    parser = argparse.ArgumentParser(description="postprocess")
    parser.add_argument('-f', '--folder', default="test", help="test data folder")
    parser.add_argument('-p', '--predfolder', default='', help="prediction folder (kosmos25)")
    parser.add_argument('-m', '--modeltype', choices=['fasterrcnn', 'kosmos25', 'tabletransformer'], default='kosmos25')
    parser.add_argument('--modelname')
    parser.add_argument('--datasetname', default="BonnData")

    parser.add_argument('--tablerelative', action='store_true', default=False)
    parser.add_argument('--no-tablerelative', dest='tablerelative', action='store_false')

    parser.add_argument('--filter', action='store_true', default=False)
    parser.add_argument('--no-filter', dest='filter', action='store_false')

    parser.add_argument('--valid_filter', action='store_true', default=False)
    parser.add_argument('--no-valid_filter', dest='valid_filter', action='store_false')

    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--no-eval', dest='eval', action='store_false')

    parser.add_argument('--iou_thresholds', nargs='*', type=float, default=[0.5, 0.6, 0.7, 0.8, 0.9])

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    dpath = f"{Path(__file__).parent.absolute()}/../../data/{args.datasetname}/{args.folder}"
    if not args.eval:
        if args.modeltype == 'fasterrcnn':
            postprocess_rcnn(modelname=args.modelname, targetloc=dpath, datasetname=args.datasetname, filter=args.filter, valid=args.valid_filter)
        elif args.modeltype == 'tabletransformer':
            postprocess_tabletransformer(modelname=args.modelname, targetloc=dpath, datasetname=args.datasetname, filter=args.filter, valid=args.valid_filter)
        elif args.modeltype == 'kosmos25':
            predpath = f"{Path(__file__).parent.absolute()}/../../results/kosmos25/{args.datasetname}{f'/{args.predfolder}' if args.predfolder else ''}"
            postprocess_kosmos(targetloc=dpath, predloc=predpath, datasetname=args.datasetname)
    else:
        postprocess_eval(datasetname=args.datasetname, modeltype=args.modeltype, modelname=args.modelname if args.modeltype in ['fasterrcnn', 'tabletransformer'] else None, tablerelative=args.tablerelative, iou_thresholds=args.iou_thresholds,
                         filter=args.filter, valid=args.valid_filter)
    exit()
    # postprocess_tabletransformer(modelname="tabletransformer_v0_new_BonnDataFullImage_tabletransformer_estest_BonnData_fullimage_e250_valid_es.pt", targetloc= f"{Path(__file__).parent.absolute()}/../../data/BonnData/test",
    #                             datasetname="BonnData")
    # postprocess_eval(modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/tabletransformer/tabletransformer_v0_new_BonnDataFullImage_tabletransformer_estest_BonnData_fullimage_e250_valid_es.pt", targetloc= f"{Path(__file__).parent.absolute()}/../../data/BonnData/test",
    #                             datasetname="BonnData", modeltype="tabletransformer")
    # postprocess_tabletransformer(
    #    modelname="tabletransformer_v0_new_GloSatFullImage_tabletransformer_newenv_fixed_GloSat_fullimage_e250_valid_es.pt",
    #    targetloc=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
    #    datasetname="GloSat")
    # postprocess_eval(
    #    modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/tabletransformer/tabletransformer_v0_new_GloSatFullImage_tabletransformer_newenv_fixed_GloSat_fullimage_e250_valid_es.pt",
    #    targetloc=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
    #    datasetname="GloSat", modeltype="tabletransformer")

    for cat in glob.glob(
        f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/*"
    ):
        # print(cat)
        # postprocess_tabletransformer(
        #    modelname="titw_call_aachen_e250_es.pt",
        #    targetloc=cat,
        #    datasetname=f"Tablesinthewild/{cat.split('/')[-1]}")
        # postprocess_eval(
        #    modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/tabletransformer/titw_call_aachen_e250_es.pt",
        #    targetloc=cat,
        #    datasetname=f"Tablesinthewild/{cat.split('/')[-1]}", modeltype="tabletransformer")
        if cat.split("/")[-1] in ["muticolorandgrid"]:
            print(cat)
            postprocess_tabletransformer(
                modelname="titw_severalcalls_2_e250_es.pt",
                targetloc=cat,
                datasetname=f"Tablesinthewild/{cat.split('/')[-1]}",
            )
            # postprocess_eval(
            #    modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/tabletransformer/titw_severalcalls_2_e250_es.pt",
            #    targetloc=cat,
            #    datasetname=f"Tablesinthewild/{cat.split('/')[-1]}", modeltype="tabletransformer")
    """
    postprocess_kosmos_sep(targetloc = f"{Path(__file__).parent.absolute()}/../../data/BonnData/test",
    predloc = f"{Path(__file__).parent.absolute()}/../../results/kosmos25/BonnData"
    f"/Tabellen/test",
    datasetname = "BonnData")

    postprocess_kosmos_sep(targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/simple",
                       predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/simple",
                       datasetname="Tablesinthewild/simple")
    postprocess_kosmos_sep(targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/curved",
                       predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/curved",
                       datasetname="Tablesinthewild/curved")
    postprocess_kosmos_sep(targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/occblu",
                       predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/occblu",
                       datasetname="Tablesinthewild/occblu")
    postprocess_kosmos_sep(targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/overlaid",
                       predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/overlaid",
                       datasetname="Tablesinthewild/overlaid")
    postprocess_kosmos_sep(targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/Inclined",
                       predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/Inclined1",
                       datasetname="Tablesinthewild/Inclined")
    postprocess_kosmos_sep(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/extremeratio",
        predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/extremeratio1",
        datasetname="Tablesinthewild/extremeratio")
    postprocess_kosmos_sep(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/muticolorandgrid",
        predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/muticolorandgrid1",
        datasetname="Tablesinthewild/muticolorandgrid")

    postprocess_kosmos_sep(targetloc=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
                       predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/GloSat/test1",
                       datasetname="GloSat")
    postprocess_rcnn_sep(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/BonnDataFullImage1_BonnData_fullimage_e250_es.pt",
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test", datasetname="BonnData")
    postprocess_rcnn_sep(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_end.pt",
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test", datasetname="BonnData")
    postprocess_rcnn_sep(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_es.pt",
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test", datasetname="BonnData")
    postprocess_rcnn_sep(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/GloSatFullImage1_GloSat_fullimage_e250_es.pt",
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test", datasetname="GloSat")
    for cat in glob.glob(f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/*"):
        print(cat)
        postprocess_rcnn_sep(targetloc=cat, datasetname=f"Tablesinthewild/{cat.split('/')[-1]}",
                         modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/testseveralcalls_4_with_valid_split_Tablesinthewild_fullimage_e50_es.pt")
        postprocess_rcnn_sep(targetloc=cat, datasetname=f"Tablesinthewild/{cat.split('/')[-1]}",
                         modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/testseveralcalls_5_without_valid_split_Tablesinthewild_fullimage_e50_end.pt")
    postprocess_kosmos(targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/simple",
                       predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/simple",
                       datasetname="Tablesinthewild/simple")
    postprocess_kosmos(targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/curved",
                       predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/curved",
                       datasetname="Tablesinthewild/curved")
    postprocess_kosmos(targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/occblu",
                       predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/occblu",
                       datasetname="Tablesinthewild/occblu")
    postprocess_kosmos(targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/overlaid",
                       predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/overlaid",
                       datasetname="Tablesinthewild/overlaid")
    postprocess_kosmos(targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/Inclined",
                       predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/Inclined1",
                       datasetname="Tablesinthewild/Inclined")
    postprocess_kosmos(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/extremeratio",
        predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/extremeratio1",
        datasetname="Tablesinthewild/extremeratio")
    postprocess_kosmos(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/muticolorandgrid",
        predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/muticolorandgrid1",
        datasetname="Tablesinthewild/muticolorandgrid")
    postprocess_kosmos()
    postprocess_kosmos(targetloc=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
                       predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/GloSat/test1",
                       datasetname="GloSat")
    # for cat in glob.glob(f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/*"):
    #    print(cat)
    #    postprocess_kosmos(targetloc=cat, predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/{cat.split('/')[-1]}", datasetname=f"Tablesinthewild/{cat.split('/')[-1]}")
    """
    """
    postprocess_rcnn(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/BonnDataFullImage1_BonnData_fullimage_e250_es.pt",
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test", datasetname="BonnData")
    postprocess_rcnn(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_end.pt",
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test", datasetname="BonnData")
    postprocess_rcnn(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_es.pt",
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test", datasetname="BonnData")
    postprocess_rcnn(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/GloSatFullImage1_GloSat_fullimage_e250_es.pt",
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test", datasetname="GloSat")
    for cat in glob.glob(f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/*"):
        print(cat)
        postprocess_rcnn(targetloc=cat, datasetname=f"Tablesinthewild/{cat.split('/')[-1]}",
                         modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/testseveralcalls_4_with_valid_split_Tablesinthewild_fullimage_e50_es.pt")
        postprocess_rcnn(targetloc=cat, datasetname=f"Tablesinthewild/{cat.split('/')[-1]}",
                         modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/testseveralcalls_5_without_valid_split_Tablesinthewild_fullimage_e50_end.pt")
    """
    # postprocess_eval(
    #    datasetname="BonnData",
    #    modeltype="kosmos25",
    #    targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test",
    # )
    """
    postprocess_eval(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/BonnDataFullImage1_BonnData_fullimage_e250_es.pt",
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test",
        datasetname="BonnData",
        modeltype="fasterrcnn",
    )
    postprocess_eval(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_end.pt",
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test",
        datasetname="BonnData",
        modeltype="fasterrcnn",
    )
    postprocess_eval(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_es.pt",
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test",
        datasetname="BonnData",
        modeltype="fasterrcnn",
    )
    """
    # postprocess_eval(
    #    datasetname="GloSat",
    #    modeltype="kosmos25",
    #    targetloc=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
    # )
    """
    postprocess_eval(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/GloSatFullImage1_GloSat_fullimage_e250_es.pt",
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
        datasetname="GloSat",
        modeltype="fasterrcnn",
    )
    """
    """
    postprocess_eval(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/simple",
        modeltype="kosmos25",
        datasetname="Tablesinthewild/simple",
    )
    postprocess_eval(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/curved",
        modeltype="kosmos25",
        datasetname="Tablesinthewild/curved",
    )
    postprocess_eval(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/occblu",
        modeltype="kosmos25",
        datasetname="Tablesinthewild/occblu",
    )
    postprocess_eval(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/overlaid",
        modeltype="kosmos25",
        datasetname="Tablesinthewild/overlaid",
    )
    postprocess_eval(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/Inclined",
        modeltype="kosmos25",
        datasetname="Tablesinthewild/Inclined",
    )
    postprocess_eval(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/extremeratio",
        modeltype="kosmos25",
        datasetname="Tablesinthewild/extremeratio",
    )
    postprocess_eval(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/muticolorandgrid",
        modeltype="kosmos25",
        datasetname="Tablesinthewild/muticolorandgrid",
    )
    """
    """
    for cat in glob.glob(
        f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/*"
    ):
        print(cat)
        postprocess_eval(
            modeltype="fasterrcnn",
            targetloc=cat,
            datasetname=f"Tablesinthewild/{cat.split('/')[-1]}",
            modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/testseveralcalls_4_with_valid_split_Tablesinthewild_fullimage_e50_es.pt",
        )
        postprocess_eval(
            modeltype="fasterrcnn",
            targetloc=cat,
            datasetname=f"Tablesinthewild/{cat.split('/')[-1]}",
            modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/testseveralcalls_5_without_valid_split_Tablesinthewild_fullimage_e50_end.pt",
        )
    postprocess_kosmos(targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/simple",
                       predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/simple",
                       datasetname="Tablesinthewild/simple")
    postprocess_kosmos(targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/curved",
                       predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/curved",
                       datasetname="Tablesinthewild/curved")
    postprocess_kosmos(targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/occblu",
                       predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/occblu",
                       datasetname="Tablesinthewild/occblu")
    postprocess_kosmos(targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/overlaid",
                       predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/overlaid",
                       datasetname="Tablesinthewild/overlaid")
    postprocess_kosmos(targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/Inclined",
                       predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/Inclined1",
                       datasetname="Tablesinthewild/Inclined")
    postprocess_kosmos(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/extremeratio",
        predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/extremeratio1",
        datasetname="Tablesinthewild/extremeratio")
    postprocess_kosmos(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/muticolorandgrid",
        predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/muticolorandgrid1",
        datasetname="Tablesinthewild/muticolorandgrid")
    postprocess_kosmos()
    postprocess_kosmos(targetloc=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
                       predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/GloSat/test1", datasetname="GloSat")
    #for cat in glob.glob(f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/*"):
    #    print(cat)
    #    postprocess_kosmos(targetloc=cat, predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/{cat.split('/')[-1]}", datasetname=f"Tablesinthewild/{cat.split('/')[-1]}")
    """
    """
    postprocess_rcnn(modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/BonnDataFullImage1_BonnData_fullimage_e250_es.pt", targetloc = f"{Path(__file__).parent.absolute()}/../../data/BonnData/test", datasetname="BonnData")
    postprocess_rcnn(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_end.pt",
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test", datasetname="BonnData")
    postprocess_rcnn(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_es.pt",
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test", datasetname="BonnData")
    postprocess_rcnn(modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/GloSatFullImage1_GloSat_fullimage_e250_es.pt", targetloc=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test", datasetname="GloSat")
    for cat in glob.glob(f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/*"):
        print(cat)
        postprocess_rcnn(targetloc=cat, datasetname=f"Tablesinthewild/{cat.split('/')[-1]}",
                          modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/testseveralcalls_4_with_valid_split_Tablesinthewild_fullimage_e50_es.pt")
        postprocess_rcnn(targetloc=cat, datasetname=f"Tablesinthewild/{cat.split('/')[-1]}",
                          modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/testseveralcalls_5_without_valid_split_Tablesinthewild_fullimage_e50_end.pt")
    """
