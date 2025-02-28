"""Evaluation for Tabletransformer."""
import argparse
import glob
import os
from pathlib import Path
from typing import Optional, List

import pandas
import torch
from lightning.fabric.utilities import move_data_to_device
from tqdm import tqdm
from transformers import (
    TableTransformerForObjectDetection,
)

from src.historicdocumentprocessing.fasterrcnn_eval import tableareabboxes
from src.historicdocumentprocessing.util.metricsutil import calcstats_iodt, calcstats_overlap, calcmetric_overlap, \
    calcstats_iou, calcmetric, get_dataframe
from src.historicdocumentprocessing.tabletransformer_dataset import CustomDataset
from src.historicdocumentprocessing.util.tablesutil import getcells


def inference(
    modelpath: Optional[str] = None,
    targetloc=None,
    iou_thresholds: Optional[List[float]] = None,
    tableareaonly=False,
    filtering: bool = False,
    valid: bool = True,
    datasetname: str = "BonnData",
    celleval: bool = False,
):
    """Inference and eval with Tabletransformer.

    Args:
        modelpath: path to model checkpoint
        targetloc: path to folder with data
        iou_thresholds: iou threshold list
        tableareaonly: wether to calculate results only for BBox-Predicitions in table area or for all predictions
        filtering: wether to filter the results
        valid: wether to use valid filter value for filtering
        datasetname: name of dataset
        celleval: wether to evaluate on cells instead of row/columns

    """
    if not iou_thresholds:
        iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    modelname = "base_table-transformer-structure-recognition"
    model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-structure-recognition"
    )
    if modelpath:
        modelname = modelpath.split("/")[-1]
        if "aachen" in modelname:
            model.load_state_dict(torch.load(modelpath, map_location="cuda:0"))
        else:
            model.load_state_dict(torch.load(modelpath))
        print("loaded_model:", modelname)
    assert model.config.id2label[1] == "table column"
    assert model.config.id2label[2] == "table row"
    # image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    # image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        model.eval()
    else:
        print("Cuda not available")
        return
    if filtering:
        with open(
            f"{Path(__file__).parent.absolute()}/../../results/tabletransformer/bestfilterthresholds{'_valid' if valid else ''}/{modelname}.txt",
            "r",
        ) as f:
            filterthreshold = float(f.read())
    saveloc = f"{Path(__file__).parent.absolute()}/../../results/tabletransformer/testevalfinal1/fullimg/{datasetname}/{modelname}/no_filtering_iou_{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}"
    # boxsaveloc = f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/{datasetname}/{modelname}"
    if tableareaonly and not filtering:
        saveloc = f"{Path(__file__).parent.absolute()}/../../results/tabletransformer/testevalfinal1/fullimg/{datasetname}/{modelname}/tableareaonly/no_filtering_iou_{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}"
    elif filtering and not tableareaonly:
        saveloc = f"{Path(__file__).parent.absolute()}/../../results/tabletransformer/testevalfinal1/fullimg/{datasetname}/{modelname}/filtering_{filterthreshold}{'_novalid' if not valid else ''}_iou{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}"
    elif filtering and tableareaonly:
        saveloc = f"{Path(__file__).parent.absolute()}/../../results/tabletransformer/testevalfinal1/fullimg/{datasetname}/{modelname}/tableareaonly/filtering_{filterthreshold}{'_novalid' if not valid else ''}_iou{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}"
    saveloc = f"{saveloc}{'/cells' if celleval else ''}"
    os.makedirs(saveloc, exist_ok=True)

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

    dataset = CustomDataset(
        targetloc,
        "fullimage",
        transforms=None,
    )
    for i in tqdm(range(len(dataset))):
        img, fullimagegroundbox, labels = dataset.getimgtarget(i, addtables=False)
        if celleval:
            fullimagegroundbox, labels = dataset.getcells(i, addtables=False)
        encoding = move_data_to_device(
            dataset.ImageProcessor(img, return_tensors="pt"), device=device
        )
        folder = dataset.getfolder(i)
        imname = folder.split("/")[-1]

        with torch.no_grad():
            results = model(**encoding)
        width, height = img.size
        if filtering:
            output = dataset.ImageProcessor.post_process_object_detection(
                results, threshold=filterthreshold, target_sizes=[(height, width)]
            )[0]
        else:
            output = dataset.ImageProcessor.post_process_object_detection(
                results, threshold=0.0, target_sizes=[(height, width)]
            )[0]
        # labels = torch.Tensor([str(model.config.id2label[label]) for label in output["labels"].tolist()])
        if celleval:
            boxes = getcells(
                rows=output["boxes"][output["labels"] == 2],
                cols=output["boxes"][output["labels"] == 1],
            )
        else:
            boxes = torch.vstack(
                [
                    output["boxes"][output["labels"] == 2],
                    output["boxes"][output["labels"] == 1],
                ]
            )
        if tableareaonly:
            boxes = tableareabboxes(boxes, folder)
        fullimagepredbox = boxes.to(device="cpu")

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

        iodtdf = pandas.concat([iodtdf, pandas.DataFrame(iodtmetrics, index=[i])])

        # .................................
        # fullimagemetrics with iou
        # .................................
        fullprediou, fulltargetiou, fulltp, fullfp, fullfn = calcstats_iou(
            fullimagepredbox,
            fullimagegroundbox,
            iou_thresholds=iou_thresholds,
            imname=imname,
        )
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
        if imname == "000d210d0f9b4101af2006b4aab33c42":
            print(fullfn)
            print(fullimagegroundbox)
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
            [overlapdf, pandas.DataFrame(overlapmetric, index=[i])]
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
            [fullimagedf, pandas.DataFrame(fullimagemetrics, index=[i])]
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
    overlapdf.to_csv(f"{saveloc}/fullimageoverlapeval.csv")
    fullimagedf.to_csv(f"{saveloc}/fullimageiou.csv")
    iodtdf.to_csv(f"{saveloc}/fullimageiodt.csv")
    conclusiondf.to_csv(f"{saveloc}/overview.csv")


def get_args() -> argparse.Namespace:
    """Define args."""   # noqa: DAR201
    parser = argparse.ArgumentParser(description="tabletransformer_eval")
    parser.add_argument('-f', '--folder', default="test", help="test data folder")
    parser.add_argument('-m', '--modelname')
    parser.add_argument('--datasetname', default="BonnData")

    parser.add_argument('--celleval', action='store_true', default=False)
    parser.add_argument('--no-celleval', dest='tablerelative', action='store_false')

    parser.add_argument('--tableareaonly', action='store_true', default=False)
    parser.add_argument('--no-tableareaonly', dest='tableareaonly', action='store_false')

    parser.add_argument('--filter', action='store_true', default=False)
    parser.add_argument('--no-filter', dest='filter', action='store_false')

    parser.add_argument('--valid_filter', action='store_true', default=False)
    parser.add_argument('--no-valid_filter', dest='valid_filter', action='store_false')

    parser.add_argument('--per_category', action='store_true', default=False)
    parser.add_argument('--no-per_category', dest='per_category', action='store_false')
    parser.add_argument('--catfolder', default="testsubclasses")

    parser.add_argument('--iou_thresholds', nargs='*', type=float, default=[0.5, 0.6, 0.7, 0.8, 0.9])

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    dpath = f"{Path(__file__).parent.absolute()}/../../data/{args.datasetname}/{args.folder}"
    mpath = f"{Path(__file__).parent.absolute()}/../../checkpoints/tabletransformer/{args.modelname}"

    if args.per_category:
        for cat in glob.glob(
                f"{Path(__file__).parent.absolute()}/../../data/{args.datasetname}/{args.catfolder}/*"
        ):
            print(cat)
            inference(modelpath=mpath, targetloc=dpath, tableareaonly=args.tableareaonly, filtering=args.filter, valid=args.valid_filter, celleval=args.celleval, iou_thresholds=args.iou_thresholds)
    else:
        inference(modelpath=mpath, targetloc=dpath, iou_thresholds=args.iou_thresholds, tableareaonly=args.tableareaonly, filtering=args.filter, valid=args.valid_filter, datasetname=args.datasetname, celleval=args.celleval)
