import glob
import json
import os
import warnings
from pathlib import Path
from typing import Tuple, List

import pandas
import torch
from torchvision.ops import box_iou
from torchvision.ops.boxes import _box_inter_union
from torchvision.ops import box_area


def reversetablerelativebboxes_inner(tablebbox: torch.Tensor, cellbboxes: torch.Tensor) -> torch.Tensor:
    """
    returns bounding boxes relative to table rather than relative to image so that evaluation of bounding box accuracy is possible on whole image
    Args:
        tablebbox:
        cellbboxes:

    Returns:

    """
    #print(cellbboxes.shape)
    return cellbboxes + torch.tensor(data=[tablebbox[0], tablebbox[1], tablebbox[0], tablebbox[1]])


def reversetablerelativebboxes_outer(fpath: str) -> torch.Tensor:
    tablebboxes = torch.load(glob.glob(f"{fpath}/*tables.pt")[0])
    #print(tablebboxes)
    newcoords = torch.zeros((0, 4))
    for table in glob.glob(f"{fpath}/*_cell_*.pt"):
        n = int(table.split(".")[-2].split("_")[-1])
        newcell = reversetablerelativebboxes_inner(tablebboxes[n], torch.load(table))
        newcoords = torch.vstack((newcoords, newcell))
    return newcoords


def extractboundingbox(bbox: dict) -> Tuple[int, int, int, int]:
    """

    Args:
        bbox: dict of bounding box coordinates

    Returns: Tuple of bounding box coordinates in format x_min, y_min, x_max, y_max

    """
    return int(bbox['x0']), int(bbox['y0']), int(bbox['x1']), int(bbox['y1'])


def boxoverlap(bbox: Tuple[int, int, int, int], tablebox: Tuple[int, int, int, int], fuzzy: int = 25) -> bool:
    """
    Checks if bbox lies in tablebox
    Args:
        bbox: the Bounding Box
        tablebox: the Bounding Box that bbox should lie in
        fuzzy: fuzzyness of tablebox boundaries
    Returns:

    """
    return bbox[0] >= tablebox[0] - fuzzy and bbox[1] >= tablebox[1] - fuzzy and bbox[2] <= tablebox[2] + fuzzy and \
        bbox[3] <= tablebox[3] + fuzzy


def extractboxes(boxdict: dict, fpath: str = None) -> torch.tensor:
    """Takes a Kosmos2.5-Output-Style Dict and extracts the bounding boxes
    Args:
        fpath: path to table folder (if only bboxes in a table wanted)
        boxdict(dict): dictionary in style of kosmos output (from json)

    Returns:
        bounding boxes as torch.tensor
    """
    boxlist = []
    tablebboxes = None
    if fpath:
        tablebboxes = torch.load(glob.glob(f"{fpath}/*tables.pt")[0])
        #print(tablebboxes)
        #print(tablebboxes)
    for box in boxdict['results']:
        bbox = extractboundingbox(box['bounding box'])
        #print(bbox)
        if fpath:
            for table in tablebboxes:
                if boxoverlap(bbox, table):
                    boxlist.append(bbox)
        else:
            boxlist.append(bbox)
    #print(boxlist)
    return torch.tensor(boxlist) if boxlist else torch.empty(0, 4)


def calcstats_IoDT(predbox: torch.tensor, targetbox: torch.tensor, imname: str = None,
                   iou_thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9]) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate tp, tp, fn based on IoDT (Intersection over Detection) at given IoU Thresholds
    Args:
        predbox:
        targetbox:
        imname:
        iou_thresholds:

    Returns:

    """
    threshold_tensor = torch.tensor(iou_thresholds)
    IoDT = Intersection_over_Detection(predbox, targetbox)
    #print(IoDT, IoDT.shape)
    prediodt = IoDT.amax(dim=1)
    targetiodt = IoDT.amax(dim=0)
    #print(predbox.shape, prediodt.shape, prediodt)
    #print(targetbox.shape, targetiodt.shape, targetiodt)
    #print(imname)
    tp = torch.sum(prediodt.unsqueeze(-1).expand(-1, len(threshold_tensor)) >= threshold_tensor, dim=0)
    # print(tp, prediodt.unsqueeze(-1).expand(-1, len(threshold_tensor)))
    fp = IoDT.shape[0] - tp
    fn = torch.sum(targetiodt.unsqueeze(-1).expand(-1, len(threshold_tensor)) < threshold_tensor, dim=0)
    #print(tp, fp, fn)
    return prediodt, targetiodt, tp, fp, fn


def Intersection_over_Detection(predbox, targetbox):
    """
    Calculate the IoDT (Intersection over Detection Metric): Intersection of prediction and target over the total prediction area
    Args:
        predbox:
        targetbox:

    Returns:

    """
    inter, union = _box_inter_union(targetbox, predbox)
    predarea = box_area(predbox)
    # print(inter, inter.shape)
    # print(predarea, predarea.shape)
    IoDT = torch.div(inter, predarea)
    # print(IoDT, IoDT.shape)
    IoDT = IoDT.T
    return IoDT


def calcstats_overlap(predbox: torch.tensor, targetbox: torch.tensor, imname: str = None, fuzzy: int = 25) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates stats based on wether a prediciton box fully overlaps with a target box instead of IoU
    tp = prediciton lies fully within a target box
    fp = prediciton does not lie fully within a target box
    fn = target boxes that do not have a prediciton lying fully within them
    Args:
        fuzzy:
        predbox:
        targetbox:
        imname:

    Returns:


    """
    #print(predbox.shape, targetbox.shape)
    if not predbox.numel():
        #print(predbox.shape)
        warnings.warn(f"A Prediction Bounding Box Tensor in {imname} is empty. (no predicitons)")
        return torch.zeros(1), torch.zeros(1), torch.full([1], fill_value=targetbox.shape[0])
    overlapmat = torch.zeros((predbox.shape[0], targetbox.shape[0]))
    for i, pred in enumerate(predbox):
        for j, target in enumerate(targetbox):
            if boxoverlap(bbox=pred, tablebox=target, fuzzy=fuzzy):
                #print(imname, pred,target)
                overlapmat[i, j] = 1
    predious = overlapmat.amax(dim=1)
    targetious = overlapmat.amax(dim=0)
    tp = torch.sum(predious.unsqueeze(-1), dim=0)
    # print(tp, predious.unsqueeze(-1).expand(-1, len(threshold_tensor)))
    fp = overlapmat.shape[0] - tp
    fn = torch.sum(targetious.unsqueeze(-1), dim=0)
    #print(overlapmat.shape, predbox.shape, targetbox.shape)
    #print(tp, fp, fn)
    #print(type(tp))
    #print(tp.shape)
    return tp, fp, fn


def calcmetric_overlap(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate Metrics from true positives, false positives, false negatives based on wether a prediciton box fully
    overlaps with a target box Args: tp: fp: fn:

    Returns:

    """
    precision = torch.nan_to_num(tp / (tp + fp))
    recall = torch.nan_to_num(tp / (tp + fn))
    f1 = torch.nan_to_num(2 * (precision * recall) / (precision + recall))
    #print(precision, recall, f1)
    return precision, recall, f1


def calcstats(predbox: torch.tensor, targetbox: torch.tensor,
              iou_thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9], imname: str = None) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the intersection over union as well as resulting tp, fp, fn at given IoU Thresholds for the bounding boxes
    of a target and prediciton given as torch tensor with bounding box
    coordinates in format x_min, y_min, x_max, y_max
    Args:
        predbox: path to predicition (json file)
        targetbox: path to target (json file)
        iou_thresholds: List of IoU Thresholds

    Returns:
        tuple of predious, targetious, tp, fp, fn

    """
    threshold_tensor = torch.tensor(iou_thresholds)
    if not predbox.numel():
        #print(predbox.shape)
        warnings.warn(f"A Prediction Bounding Box Tensor in {imname} is empty. (no predicitons)")
        return torch.zeros(predbox.shape[0]), torch.zeros(targetbox.shape[0]), torch.zeros(
            threshold_tensor.shape), torch.zeros(threshold_tensor.shape), torch.full(threshold_tensor.shape,
                                                                                     fill_value=targetbox.shape[0])
    ioumat = box_iou(predbox, targetbox)
    predious = ioumat.amax(dim=1)
    targetious = ioumat.amax(dim=0)

    #print(predious)
    #print(targetious)
    #mine
    tp = torch.sum(predious.unsqueeze(-1).expand(-1, len(threshold_tensor)) >= threshold_tensor, dim=0)
    #print(tp, predious.unsqueeze(-1).expand(-1, len(threshold_tensor)))
    fp = ioumat.shape[0] - tp
    fn = torch.sum(targetious.unsqueeze(-1).expand(-1, len(threshold_tensor)) < threshold_tensor, dim=0)
    #print(tp, fp, fn, ioumat.shape)

    #from our project group
    # n_pred, n_target = ioumat.shape
    #tp = torch.sum(
    #    predious.expand(len(iou_thresholds), n_pred) >= threshold_tensor[:, None], dim=1)
    #fp = len(ioumat) - tp
    #fn = torch.sum(
    #    targetious.expand(len(iou_thresholds), n_target) < threshold_tensor[:, None], dim=1)
    return predious, targetious, tp, fp, fn


def calcmetric(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor,
               iou_thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9]) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate Metrics from true positives, false positives, false negatives at given iou thresholds
    Args:
        tp:
        fp:
        fn:
        iou_thresholds:

    Returns:

    """
    threshold_tensor = torch.tensor(iou_thresholds)
    precision = torch.nan_to_num(tp / (tp + fp))
    recall = torch.nan_to_num(tp / (tp + fn))
    f1 = torch.nan_to_num(2 * (precision * recall) / (precision + recall))
    wf1 = f1 @ threshold_tensor / torch.sum(threshold_tensor)
    return precision, recall, f1, wf1


def calcmetrics_engnewspaper(targetloc: str = f"{Path(__file__).parent.absolute()}/../../data/EngNewspaper/Kosmos",
                             predloc: str = f"{Path(__file__).parent.absolute()}/../../results/kosmos25/EngNewspaper",
                             iou_thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9]) -> Tuple[
    List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates metrics for all predictions and corresponding targets in a given location (Bboxes given as json files)
    !incomplete since it wasnt needed so far!
    Args:
        targetloc: target location
        predloc: prediction location

    Returns: nested list of all ious

    """
    preds = glob.glob(f"{predloc}/*.json")
    ioulist = []
    f1list = []
    wf1list = []
    tpsum = torch.zeros(len(iou_thresholds))
    fpsum = torch.zeros(len(iou_thresholds))
    fnsum = torch.zeros(len(iou_thresholds))
    for pred in preds:
        target = glob.glob(f"{targetloc}/{pred.split('/')[-1].split('.')[-3]}.json")
        #print(target, pred)
        if target:
            with open(pred) as p:
                predbox = extractboxes(json.load(p))
            # print(predbox)
            with open(target[0]) as t:
                targetbox = extractboxes(json.load(t))
            # print(targetbox)
            iou, tp, fp, fn = calcstats(predbox, targetbox, iou_thresholds=iou_thresholds)
            prec, rec, f1, wf1 = calcmetric(tp, fp, fn, iou_thresholds=iou_thresholds)
            tpsum += tp
            fpsum += fp
            fnsum += fn
            ioulist.append(iou)
            f1list.append(f1)
            wf1list.append(wf1)
    totalprec, totalrec, totalf1, totalwf1 = calcmetric(tpsum, fpsum, fnsum)
    return ioulist, f1list, wf1list, totalprec, totalrec, totalf1, totalwf1


def calcmetrics_tables(targetloc: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/test",
                       predloc: str = f"{Path(__file__).parent.absolute()}/../../results/kosmos25/BonnData"
                                      f"/Tabellen/test",
                       iou_thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9],
                       saveloc: str = f"{Path(__file__).parent.absolute()}/../../results/kosmos25/BonnData/Tabellen"
                                      f"/testeval",
                       tableavail: bool = True) -> \
        Tuple[
            List[torch.Tensor], List[torch.Tensor], List[
                torch.Tensor]]:
    """
    Calculates metrics for all predictions and corresponding targets in a given location (pred Bboxes given as json
    files, targets as torch.Tensor) Args: targetloc: target location predloc: prediction location

    Returns: nested list of all ious

    """
    predfolder = glob.glob(f"{predloc}/*")
    #fullimagevars
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
    #tablevars
    ioulist = []
    f1list = []
    wf1list = []
    predcount = 0
    tpsum = torch.zeros(len(iou_thresholds))
    fpsum = torch.zeros(len(iou_thresholds))
    fnsum = torch.zeros(len(iou_thresholds))
    overlapdf = pandas.DataFrame(columns=["img", "prednum"])
    fullimagedf = pandas.DataFrame(columns=["img", "mean_pred_iou", "mean_tar_iou", "wf1", "prednum"])
    tabledf = pandas.DataFrame(columns=["img", "mean_pred_iou", "mean_tar_iou", "wf1", "prednum"])
    for n, preds in enumerate(predfolder):
        targets = glob.glob(f"{targetloc}/{preds.split('/')[-1]}")
        #imagepred = [file for file in glob.glob(f"{preds}/*.json") if "_table_" not in file][0]
        #bboxes on full image
        #fullimagepred = glob.glob(f"{preds}/(?!.*_table_)^.*$")[0]
        fullimagepred = [file for file in glob.glob(f"{preds}/*") if "_table_" not in file][0]
        #print(fullimagepred)
        with open(fullimagepred) as p:
            fullimagepredbox = extractboxes(json.load(p), fpath=targets[0] if tableavail else None)
            #print(targets[0])

        if tableavail:
            fullimagegroundbox = reversetablerelativebboxes_outer(targets[0])
        else:
            fullimagegroundbox = torch.load(glob.glob(f"{targetloc}/{preds.split('/')[-1]}/*pt")[0])

        _, _, IoDT_tp, IoDT_fp, IoDT_fn = calcstats_IoDT(predbox=fullimagepredbox, targetbox=fullimagegroundbox,
                                                         imname=preds.split('/')[-1])
        print(IoDT_tp, IoDT_fp, IoDT_fn)
        pass

        # .................................
        # fullimagemetrics with iou
        # .................................
        fullprediou, fulltargetiou, fulltp, fullfp, fullfn = calcstats(fullimagepredbox, fullimagegroundbox,
                                                                       iou_thresholds=iou_thresholds,
                                                                       imname=preds.split('/')[-1])
        #print(calcstats(fullimagepredbox, fullimagegroundbox,
        #                                                               iou_thresholds=iou_thresholds, imname=preds.split('/')[-1]), fullfp, targets[0])
        fullprec, fullrec, fullf1, fullwf1 = calcmetric(fulltp, fullfp, fullfn, iou_thresholds=iou_thresholds)
        fulltpsum += fulltp
        fullfpsum += fullfp
        fullfnsum += fullfn
        fullioulist.append(fullprediou)
        fullf1list.append(fullf1)
        fullwf1list.append(fullwf1)
        imname = fullimagepred.split("/")[-1].split(".")[-3]
        fullimagemetrics = {"img": imname, "mean_pred_iou": torch.mean(fullprediou).item(),
                            "mean_tar_iou": torch.mean(fulltargetiou).item(), "wf1": fullwf1.item(),
                            "prednum": fullimagepredbox.shape[0]}
        fullimagemetrics.update({f"prec_{iou_thresholds[i]}": fullprec[i].item() for i in range(len(iou_thresholds))})
        fullimagemetrics.update({f"recall_{iou_thresholds[i]}": fullrec[i].item() for i in range(len(iou_thresholds))})
        fullimagemetrics.update({f"f1_{iou_thresholds[i]}": fullf1[i].item() for i in range(len(iou_thresholds))})
        fullimagemetrics.update({f"tp_{iou_thresholds[i]}": fulltp[i].item() for i in range(len(iou_thresholds))})
        fullimagemetrics.update({f"fp_{iou_thresholds[i]}": fullfp[i].item() for i in range(len(iou_thresholds))})
        fullimagemetrics.update({f"fn_{iou_thresholds[i]}": fullfn[i].item() for i in range(len(iou_thresholds))})

        # ........................................
        # fullimagemetrics with alternate metric
        # ........................................
        fulltp_overlap, fullfp_overlap, fullfn_overlap = calcstats_overlap(fullimagepredbox, fullimagegroundbox,
                                                                           imname=preds.split('/')[-1])

        fullprec_overlap, fullrec_overlap, fullf1_overlap = calcmetric_overlap(tp=fulltp_overlap, fp=fullfp_overlap,
                                                                               fn=fullfn_overlap)
        tpsum_overlap += fulltp_overlap
        fpsum_overlap += fullfp_overlap
        fnsum_overlap += fullfn_overlap
        overlapmetric = {"img": imname, "prednum": fullimagepredbox.shape[0], f"prec": fullprec_overlap.item(),
                         f"recall": fullrec_overlap.item(), f"f1": fullf1_overlap.item(),
                         "tp": fulltp_overlap, "fp": fullfp_overlap, "fn": fullfn_overlap}

        overlapdf = pandas.concat([overlapdf, pandas.DataFrame(overlapmetric, index=[n])])

        if fullimagepredbox.shape[0] != 0:
            fulltpsum_predonly += fulltp
            fullfpsum_predonly += fullfp
            fullfnsum_predonly += fullfn
            tpsum_overlap_predonly += fulltp_overlap
            fpsum_overlap_predonly += fullfp_overlap
            fnsum_overlap_predonly += fullfn_overlap
            predcount += 1

            #print("here",fullimagepredbox)
        #print(fullfp, fullimagemetrics)
        fullimagedf = pandas.concat([fullimagedf, pandas.DataFrame(fullimagemetrics, index=[n])])
        #print(fullimagedf.loc[n])

        # ...........................
        # bboxes on tables
        # ...........................
        tablesground = glob.glob(f"{targets[0]}/*_cell_*.pt")
        #print(f"{targets[0]}/*_cell_*.pt")
        for tableground in tablesground:
            num = tableground.split(".")[-2].split("_")[-1]
            tablepred = glob.glob(f"{preds}/*_table_{num}*.json")[0]
            tablebox = extractboxes(json.load(open(tablepred)))
            #print(tablebox)
            prediou, targetiou, tp, fp, fn = calcstats(tablebox, torch.load(tableground), iou_thresholds=iou_thresholds,
                                                       imname=preds.split('/')[-1] + "_" + num)
            prec, rec, f1, wf1 = calcmetric(tp, fp, fn, iou_thresholds=iou_thresholds)
            #print(wf1)
            tpsum += tp
            fpsum += fp
            fnsum += fn
            ioulist.append(prediou)
            f1list.append(f1)
            wf1list.append(wf1)
            tablemetrics = {"img": imname, "mean_pred_iou": torch.mean(prediou).item(),
                            "mean_tar_iou": torch.mean(targetiou).item(), "wf1": wf1.item(),
                            "prednum": tablebox.shape[0]}
            tablemetrics.update(
                {f"prec_{iou_thresholds[i]}": prec[i].item() for i in range(len(iou_thresholds))})
            tablemetrics.update(
                {f"recall_{iou_thresholds[i]}": rec[i].item() for i in range(len(iou_thresholds))})
            tablemetrics.update({f"f1_{iou_thresholds[i]}": f1[i].item() for i in range(len(iou_thresholds))})
            tablemetrics.update({f"tp_{iou_thresholds[i]}": tp[i].item() for i in range(len(iou_thresholds))})
            tablemetrics.update({f"fp_{iou_thresholds[i]}": fp[i].item() for i in range(len(iou_thresholds))})
            tablemetrics.update({f"fn_{iou_thresholds[i]}": fn[i].item() for i in range(len(iou_thresholds))})
            tabledf = pandas.concat([tabledf, pandas.DataFrame(tablemetrics, index=[f"{n}.{num}"])])
            #print(tablebox.shape, tablemetrics)
            #print(tablemetrics)
    #print(fullimagedf)
    #print(tabledf)
    #totalprec, totalrec, totalf1, totalwf1 = calcmetric(tpsum, fpsum, fnsum)
    totalfullmetrics = get_dataframe(fullfnsum, fullfpsum, fulltpsum, nopredcount=fullimagedf.shape[0] - predcount,
                                     imnum=fullimagedf.shape[0], iou_thresholds=iou_thresholds)
    partialfullmetrics = get_dataframe(fullfnsum_predonly, fullfpsum_predonly, fulltpsum_predonly,
                                       iou_thresholds=iou_thresholds)
    totalmetrics = get_dataframe(fnsum, fpsum, tpsum, iou_thresholds=iou_thresholds)
    #print(totalfullmetrics)
    overlapprec, overlaprec, overlapf1 = calcmetric_overlap(tp=tpsum_overlap, fp=fpsum_overlap, fn=fnsum_overlap)
    totaloverlapdf = pandas.DataFrame({"f1": overlapf1, "prec": overlaprec, "recall": overlaprec}, index=["overlap"])
    overlapprec_predonly, overlaprec_predonly, overlapf1_predonly = calcmetric_overlap(tp=tpsum_overlap_predonly,
                                                                                       fp=fpsum_overlap_predonly,
                                                                                       fn=fnsum_overlap_predonly)
    predonlyoverlapdf = pandas.DataFrame({f"Number of evaluated files": overlapdf.shape[0],
                                          f"Evaluated files without predictions:": overlapdf.shape[0] - predcount,
                                          "f1": overlapf1_predonly, "prec": overlaprec_predonly,
                                          "recall": overlaprec_predonly}, index=["overlap with valid preds"])
    conclusiondf = pandas.DataFrame(columns=["wf1"])
    #totalmetrics = {"wf1": totalwf1.item()}
    #totalmetrics.update({f"f1_{iou_thresholds[i]}": totalf1[i].item() for i in range(len(iou_thresholds))})
    #totalmetrics.update({f"prec_{iou_thresholds[i]}": totalprec[i].item() for i in range(len(iou_thresholds))})
    #totalmetrics.update({f"recall_{iou_thresholds[i]}": totalrec[i].item() for i in range(len(iou_thresholds))})
    #totalmetrics.update({f"tp_{iou_thresholds[i]}": tpsum[i].item() for i in range(len(iou_thresholds))})
    #totalmetrics.update({f"fp_{iou_thresholds[i]}": fpsum[i].item() for i in range(len(iou_thresholds))})
    #totalmetrics.update({f"fn_{iou_thresholds[i]}": fnsum[i].item() for i in range(len(iou_thresholds))})

    conclusiondf = pandas.concat([conclusiondf, pandas.DataFrame(totalfullmetrics, index=["full image metrics"]),
                                  pandas.DataFrame(partialfullmetrics, index=["full image with valid preds"]),
                                  pandas.DataFrame(totalmetrics, index=["table metrics"]), totaloverlapdf,
                                  predonlyoverlapdf])

    #print(conclusiondf)
    #save results
    os.makedirs(saveloc, exist_ok=True)
    #print(fullimagedf.loc[50])
    overlapdf.to_csv(f"{saveloc}/fullimageoverlapeval.csv")
    fullimagedf.to_csv(f"{saveloc}/fullimageeval.csv")
    tabledf.to_csv(f"{saveloc}/tableeval.csv")
    conclusiondf.to_csv(f"{saveloc}/overview.csv")
    return fullioulist, fullf1list, fullwf1list


def get_dataframe(fnsum, fpsum, tpsum, nopredcount: int = None, imnum: int = None,
                  iou_thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9]):
    totalfullprec, totalfullrec, totalfullf1, totalfullwf1 = calcmetric(tpsum, fpsum, fnsum)
    totalfullmetrics = {"wf1": totalfullwf1.item()}
    totalfullmetrics.update({f"Number of evaluated files": imnum})
    totalfullmetrics.update({f"Evaluated files without predictions:": nopredcount})
    totalfullmetrics.update({f"f1_{iou_thresholds[i]}": totalfullf1[i].item() for i in range(len(iou_thresholds))})
    totalfullmetrics.update({f"prec_{iou_thresholds[i]}": totalfullprec[i].item() for i in range(len(iou_thresholds))})
    totalfullmetrics.update({f"recall_{iou_thresholds[i]}": totalfullrec[i].item() for i in range(len(iou_thresholds))})
    totalfullmetrics.update({f"tp_{iou_thresholds[i]}": tpsum[i].item() for i in range(len(iou_thresholds))})
    totalfullmetrics.update({f"fp_{iou_thresholds[i]}": fpsum[i].item() for i in range(len(iou_thresholds))})
    totalfullmetrics.update({f"fn_{iou_thresholds[i]}": fnsum[i].item() for i in range(len(iou_thresholds))})
    return totalfullmetrics


if __name__ == '__main__':
    #print(calcmetrics_jsoninput())
    #calcmetrics_tables(
    #    saveloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/BonnData/Tabellen/testeval/trial5")
    #calcmetrics_tables(targetloc=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
    #                   predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/GloSat/test1",
    #                   iou_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9],
    #                   saveloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/GloSat/testeval/test1")
    calcmetrics_tables(targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/simple",
                       predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/simple",
                       iou_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9],
                       saveloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/testeval/simple/withIoDT",
                       tableavail=False)
    #calcmetrics_tables(targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/curved",
    #                   predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/curved",
    #                   iou_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9],
    #                   saveloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/testeval/curved",
    #                   tableavail=False)
    #print(reversetablerelativebboxes_outer(f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/preprocessed/I_HA_Rep_89_Nr_16160_0090"))
    pass
