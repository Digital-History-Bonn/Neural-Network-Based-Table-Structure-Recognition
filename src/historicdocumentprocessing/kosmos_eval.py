"""Evaluation for Kosmos 2.5."""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas
import torch
from tqdm import tqdm

from src.historicdocumentprocessing.util.metricsutil import (
    calcmetric,
    calcmetric_overlap,
    calcstats_iodt,
    calcstats_iou,
    calcstats_overlap,
    get_dataframe,
)
from src.historicdocumentprocessing.util.tablesutil import (
    extractboxes,
    reversetablerelativebboxes_outer,
)


def calcmetrics_tables(
    targetloc: str,  # f"{Path(__file__).parent.absolute()}/../../data/BonnData/test"
    predloc: str,  # f"{Path(__file__).parent.absolute()}/../../results/kosmos25/BonnData/Tabellen/test"
    iou_thresholds: Optional[List[float]] = None,
    saveloc: Optional[str] = None,
    tablerelative: bool = True,
    tableareaonly: bool = True,
    datasetname: Optional[str] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Calculates metrics for all predictions and corresponding targets in a given location (pred Bboxes given as json files, targets as torch.Tensor).

    Args:
        targetloc: target location
        predloc: prediction location
        iou_thresholds: iou thresholds
        saveloc: save location
        tablerelative: wether to use tablerelative groundtruth BBoxes (since GloSAT and BonnData do not have image relative)
        tableareaonly: wether to calculate results only for BBox-Predicitions in table area or for all predictions
        datasetname: name of the dataset (for save locations)

    Returns:
        tuple of lists with iou, f1, wf1 values
    """
    # tableareaonly= False
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    if saveloc is None:
        saveloc = f"{Path(__file__).parent.absolute()}/../../results/kosmos25/eval/fullimg/{datasetname}"
    if tableareaonly:
        saveloc = f"{saveloc}/tableareaonly"
    saveloc = (
        f"{saveloc}/iou_{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}"
    )
    # fullimagevars
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
    # tablevars
    ioulist = []
    f1list = []
    wf1list = []
    predcount = 0
    tpsum = torch.zeros(len(iou_thresholds))
    fpsum = torch.zeros(len(iou_thresholds))
    fnsum = torch.zeros(len(iou_thresholds))
    overlapdf = pandas.DataFrame(columns=["img", "prednum"])
    fullimagedf = pandas.DataFrame(
        columns=["img", "mean pred iou", "mean tar iou", "wf1", "prednum"]
    )
    iodtdf = pandas.DataFrame(
        columns=["img", "mean pred iod", "mean tar iod", "wf1", "prednum"]
    )
    tabledf = pandas.DataFrame(
        columns=["img", "mean pred iou", "mean tar iou", "wf1", "prednum"]
    )
    for n, targets in tqdm(enumerate(glob.glob(f"{targetloc}/*"))):
        preds = glob.glob(f"{predloc}/{targets.split('/')[-1]}")[0]
        # bboxes on full image
        fullimagepred = [
            file for file in glob.glob(f"{preds}/*") if "_table_" not in file
        ][0]
        # print(fullimagepred)
        with open(fullimagepred) as p:
            fullimagepredbox = extractboxes(
                json.load(p), fpath=targets if tableareaonly else None
            )

        if tablerelative:
            fullimagegroundbox = reversetablerelativebboxes_outer(targets)
        else:
            fullimagegroundbox = torch.load(
                glob.glob(
                    f"{targetloc}/{preds.split('/')[-1]}/{preds.split('/')[-1]}.pt"
                )[0]
            )

        imname = preds.split("/")[-1]

        # .......................................
        # fullimagemetrics with IoDT
        # .......................................
        prediod, tariod, iodt_tp, iodt_fp, iodt_fn = calcstats_iodt(
            predbox=fullimagepredbox,
            targetbox=fullimagegroundbox,
            imname=preds.split("/")[-1],
            iou_thresholds=iou_thresholds,
        )
        # print(IoDT_tp, IoDT_fp, IoDT_fn)
        iodt_prec, iodt_rec, iodt_f1, iodt_wf1 = calcmetric(
            tp=iodt_tp, fp=iodt_fp, fn=iodt_fn, iou_thresholds=iou_thresholds
        )
        # print(IoDT_prec, IoDT_rec, IoDT_f1, IoDT_wf1)
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
            imname=preds.split("/")[-1],
        )
        # print(calcstats(fullimagepredbox, fullimagegroundbox,
        #                                                               iou_thresholds=iou_thresholds, imname=preds.split('/')[-1]), fullfp, targets[0])
        fullprec, fullrec, fullf1, fullwf1 = calcmetric(
            tp=fulltp, fp=fullfp, fn=fullfn, iou_thresholds=iou_thresholds
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
            fullimagepredbox, fullimagegroundbox, imname=preds.split("/")[-1]
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
        # ...........................
        # bboxes on tables
        # ...........................
        tablesground = glob.glob(f"{targets}/*_cell_*.pt")
        for tableground in tablesground:
            num = tableground.split(".")[-2].split("_")[-1]
            if glob.glob(f"{preds}/*_table_{num}*.json"):
                tablepred = glob.glob(f"{preds}/*_table_{num}*.json")[0]
                tablebox = extractboxes(json.load(open(tablepred)))
                prediou, targetiou, tp, fp, fn = calcstats_iou(
                    tablebox,
                    torch.load(tableground),
                    iou_thresholds=iou_thresholds,
                    imname=preds.split("/")[-1] + "_" + num,
                )
                prec, rec, f1, wf1 = calcmetric(
                    tp=tp, fp=fp, fn=fn, iou_thresholds=iou_thresholds
                )
                tpsum += tp
                fpsum += fp
                fnsum += fn
                ioulist.append(prediou)
                f1list.append(f1)
                wf1list.append(wf1)
                tablemetrics = {
                    "img": imname,
                    "mean pred iou": torch.mean(prediou).item(),
                    "mean tar iou": torch.mean(targetiou).item(),
                    "wf1": wf1.item(),
                    "prednum": tablebox.shape[0],
                }
                tablemetrics.update(
                    {
                        f"prec@{iou_thresholds[i]}": prec[i].item()
                        for i in range(len(iou_thresholds))
                    }
                )
                tablemetrics.update(
                    {
                        f"recall@{iou_thresholds[i]}": rec[i].item()
                        for i in range(len(iou_thresholds))
                    }
                )
                tablemetrics.update(
                    {
                        f"f1@{iou_thresholds[i]}": f1[i].item()
                        for i in range(len(iou_thresholds))
                    }
                )
                tablemetrics.update(
                    {
                        f"tp@{iou_thresholds[i]}": tp[i].item()
                        for i in range(len(iou_thresholds))
                    }
                )
                tablemetrics.update(
                    {
                        f"fp@{iou_thresholds[i]}": fp[i].item()
                        for i in range(len(iou_thresholds))
                    }
                )
                tablemetrics.update(
                    {
                        f"fn@{iou_thresholds[i]}": fn[i].item()
                        for i in range(len(iou_thresholds))
                    }
                )
                tabledf = pandas.concat(
                    [tabledf, pandas.DataFrame(tablemetrics, index=[f"{n}.{num}"])]
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
    totalmetrics = get_dataframe(
        fnsum=fnsum, fpsum=fpsum, tpsum=tpsum, iou_thresholds=iou_thresholds
    )
    overlapprec, overlaprec, overlapf1 = calcmetric_overlap(
        tp=tpsum_overlap, fp=fpsum_overlap, fn=fnsum_overlap
    )
    totaloverlapdf = pandas.DataFrame(
        {"f1": overlapf1, "prec": overlapprec, "recall": overlaprec}, index=["overlap"]
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
            pandas.DataFrame(totalmetrics, index=["table IoU"]),
            totaloverlapdf,
            predonlyoverlapdf,
            pandas.DataFrame(totaliodt, index=["full image IoDt"]),
            pandas.DataFrame(predonlyiodt, index=[" full image IoDt with valid preds"]),
        ]
    )
    # save results
    os.makedirs(saveloc, exist_ok=True)
    overlapdf.to_csv(f"{saveloc}/fullimageoverlapeval.csv")
    fullimagedf.to_csv(f"{saveloc}/fullimageiou.csv")
    tabledf.to_csv(f"{saveloc}/tableiou.csv")
    iodtdf.to_csv(f"{saveloc}/fullimageiodt.csv")
    conclusiondf.to_csv(f"{saveloc}/overview.csv")
    return fullioulist, fullf1list, fullwf1list


def get_args() -> argparse.Namespace:
    """Define args."""  # noqa: DAR201
    parser = argparse.ArgumentParser(description="kosmos_eval")
    parser.add_argument("-t", "--testfolder", default="test", help="test data folder")
    parser.add_argument("-p", "--predfolder", default="", help="prediction folder")
    parser.add_argument("--datasetname", default="BonnData")
    parser.add_argument(
        "--iou_thresholds", nargs="*", type=float, default=[0.5, 0.6, 0.7, 0.8, 0.9]
    )

    parser.add_argument("--tableareaonly", action="store_true", default=False)
    parser.add_argument(
        "--no-tableareaonly", dest="tableareaonly", action="store_false"
    )

    parser.add_argument("--tablerelative", action="store_true", default=False)
    parser.add_argument(
        "--no-tablerelative", dest="tablerelative", action="store_false"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    targetpath = f"{Path(__file__).parent.absolute()}/../../data/{args.datasetname}/{args.folder}"
    predpath = f"{Path(__file__).parent.absolute()}/../../results/kosmos25/{args.datasetname}{f'/{args.predfolder}' if args.predfolder else ''}"
    saveloc = f"{Path(__file__).parent.absolute()}/../../results/kosmos25/eval/fullimg/{args.datasetname}"

    calcmetrics_tables(
        targetloc=targetpath,
        predloc=predpath,
        iou_thresholds=args.iou_thresholds,
        saveloc=saveloc,
        tablerelative=args.tablerelative,
        tableareaonly=args.tableareaonly,
        datasetname=args.datasetname,
    )

    exit()

    calcmetrics_tables(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/test",
        predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/complete",
        iou_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9],
        saveloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/testevalfinal2/fullimg"
        f"/Tablesinthewild/allsubsets/tableavail/tablerelative/tableareaonly",
    )
    calcmetrics_tables(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/test",
        predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/complete",
        iou_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9],
        saveloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/testevalfinal2/fullimg"
        f"/Tablesinthewild/allsubsets/tableavail/tablerelative/nottableareaonly",
        tableareaonly=False,
    )
    """
    calcmetrics_tables(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/simple",
        predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/simple",
        iou_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9],
        saveloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/testevalchange/fullimg"
        f"/Tablesinthewild/simple/tableavail",
    )
    calcmetrics_tables(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/curved",
        predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/curved",
        iou_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9],
        saveloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/testevalchange/fullimg"
        f"/Tablesinthewild/curved/tableavail",
    )
    calcmetrics_tables(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/occblu",
        predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/occblu",
        iou_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9],
        saveloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/testevalchange/fullimg"
        f"/Tablesinthewild/occblu/tableavail",
    )
    calcmetrics_tables(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/overlaid",
        predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/overlaid",
        iou_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9],
        saveloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/testevalchange/fullimg"
        f"/Tablesinthewild/overlaid/tableavail",
    )
    calcmetrics_tables(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/Inclined",
        predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/Inclined1",
        iou_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9],
        saveloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/testevalchange/fullimg"
        f"/Tablesinthewild/Inclined/tableavail",
    )
    calcmetrics_tables(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/extremeratio",
        predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/extremeratio1",
        iou_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9],
        saveloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/testevalchange/fullimg"
        f"/Tablesinthewild/extremeratio/tableavail",
    )
    calcmetrics_tables(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/muticolorandgrid",
        predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/muticolorandgrid1",
        iou_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9],
        saveloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/testevalchange/fullimg"
        f"/Tablesinthewild/muticolorandgrid/tableavail",
    )
    # print(calcmetrics_jsoninput())
    calcmetrics_tables(
        saveloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/testevalchange/fullimg/BonnData"
    )
    calcmetrics_tables(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
        predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/GloSat/test1",
        iou_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9],
        saveloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/testevalchange/fullimg/GloSat",
    )
    calcmetrics_tables(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/simple",
        predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/simple",
        iou_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9],
        saveloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/testevalchange/fullimg"
        f"/Tablesinthewild/simple",
        tablerelative=False,
    )
    calcmetrics_tables(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/curved",
        predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/curved",
        iou_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9],
        saveloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/testevalchange/fullimg"
        f"/Tablesinthewild/curved",
        tablerelative=False,
    )
    calcmetrics_tables(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/occblu",
        predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/occblu",
        iou_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9],
        saveloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/testevalchange/fullimg"
        f"/Tablesinthewild/occblu",
        tablerelative=False,
    )
    calcmetrics_tables(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/overlaid",
        predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/overlaid",
        iou_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9],
        saveloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/testevalchange/fullimg"
        f"/Tablesinthewild/overlaid",
        tablerelative=False,
    )
    calcmetrics_tables(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/Inclined",
        predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/Inclined1",
        iou_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9],
        saveloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/testevalchange/fullimg"
        f"/Tablesinthewild/Inclined",
        tablerelative=False,
    )
    calcmetrics_tables(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/extremeratio",
        predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/extremeratio1",
        iou_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9],
        saveloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/testevalchange/fullimg"
        f"/Tablesinthewild/extremeratio",
        tablerelative=False,
    )
    calcmetrics_tables(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/muticolorandgrid",
        predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/muticolorandgrid1",
        iou_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9],
        saveloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/testevalchange/fullimg"
        f"/Tablesinthewild/muticolorandgrid",
        tablerelative=False,
    )
    """
    # print(reversetablerelativebboxes_outer(f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/preprocessed/I_HA_Rep_89_Nr_16160_0090"))
    pass
