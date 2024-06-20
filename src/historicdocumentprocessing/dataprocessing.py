import os
from pathlib import Path
import glob
import json
from typing import Tuple, List

import pandas
import torch
from torch import Tensor
from tqdm import tqdm
import warnings

from torchvision.ops import box_iou


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
            dicts.append({"file_name": str(datapath.split('/')[-1].split('.')[-2]), "ground_truth": {
                "gt_parse": {"newspaper": [{str(k['class']): str(k['raw_text'])} for k in string['bboxes'] if
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


def calcstats(predbox: torch.tensor, targetbox: torch.tensor,
              iou_thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9], imname: str = None) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the stats for the bounding boxes of a target and prediciton given as torch tensor with bounding box
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

    print(predious)
    print(targetious)
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
                       saveloc: str = f"{Path(__file__).parent.absolute()}/../../results/kosmos25/BonnData/Tabellen/testeval") -> \
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
    #tablevars
    ioulist = []
    f1list = []
    wf1list = []
    predcount = 0
    tpsum = torch.zeros(len(iou_thresholds))
    fpsum = torch.zeros(len(iou_thresholds))
    fnsum = torch.zeros(len(iou_thresholds))
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
            fullimagepredbox = extractboxes(json.load(p), fpath=targets[0])
            #print(targets[0])
        fullimagegroundbox = reversetablerelativebboxes_outer(targets[0])
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

        if fullimagepredbox.shape[0] != 0:
            fulltpsum_predonly += fulltp
            fullfpsum_predonly += fullfp
            fullfnsum_predonly += fullfn
            predcount += 1

            #print("here",fullimagepredbox)
        #print(fullfp, fullimagemetrics)
        fullimagedf = pandas.concat([fullimagedf, pandas.DataFrame(fullimagemetrics, index=[n])])
        #print(fullimagedf.loc[n])
        #bboxes on tables
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
                                  pandas.DataFrame(totalmetrics, index=["table metrics"])])

    #print(conclusiondf)
    #save results
    os.makedirs(saveloc, exist_ok=True)
    #print(fullimagedf.loc[50])
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


def savetotalmetrics(ioulist: List[torch.Tensor], f1list: List[torch.Tensor], wf1list: List[torch.Tensor],
                     totalprec: torch.Tensor, totalrec: torch.Tensor, totalf1: torch.Tensor, totalwf1: torch.Tensor):
    return


def main():
    pass


if __name__ == '__main__':
    #processdata_engnews_kosmos()
    processdata_engnews_donut()
    #print(calcmetrics_jsoninput())
    #calcmetrics_tables(
    #    saveloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/BonnData/Tabellen/testeval/trial4")
    #print(reversetablerelativebboxes_outer(f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/preprocessed/I_HA_Rep_89_Nr_16160_0090"))
