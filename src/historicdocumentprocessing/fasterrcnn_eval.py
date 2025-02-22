"""Inference and Evaluation for Faster RCNN."""

import glob
import os
from pathlib import Path
from typing import List

import pandas
import torch
from torchvision.io import read_image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from tqdm import tqdm

from src.historicdocumentprocessing.kosmos_eval import (
    boxoverlap,
    calcmetric,
    calcmetric_overlap,
    calcstats_iodt,
    calcstats_iou,
    calcstats_overlap,
    get_dataframe,
    reversetablerelativebboxes_outer,
)


def tableareabboxes(bboxes: torch.tensor, tablepath: str) -> torch.tensor:
    """Find the BBoxes that lie within one of the tables of an image.

    Args:
        bboxes: BoundingBox Tensor
        tablepath: path to file with table coordinates

    Returns:
        BBoxes in tables

    """
    bboxlist = []
    tablebboxes = torch.load(glob.glob(f"{tablepath}/*tables.pt")[0])
    for bbox in bboxes:
        for tablebbox in tablebboxes:
            if boxoverlap(bbox, tablebbox):
                bboxlist.append(bbox)
    return torch.vstack(bboxlist) if bboxlist else torch.empty(0, 4)


def inference_fullimg(
    targetloc: str,  # f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/test"
    modelpath: str,  # f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/test4_Tablesinthewild_fullimage_e50_end.pt"
    datasetname: str = "Tablesinthewild",
    iou_thresholds: List[float] = None,  # [0.5, 0.6, 0.7, 0.8, 0.9]
    filter: bool = True,
    tablerelative: bool = False,
    tableareaonly: bool = False,
    valid: bool = True,
):
    """Inference on the full image.

    Args:
        targetloc: folder with preprocessed test data (images and BBoxes)
        modelpath: path to model checkpoint
        datasetname: name of the dataset (for save locations)
        iou_thresholds: iou threshold list
        filter: wether to filter the results
        tablerelative: wether to use tablerelative groundtruth BBoxes (since GloSAT and BonnData do not have image relative)
        tableareaonly: wether to calculate results only for BBox-Predicitions in table area or for all predictions
        valid: wether to use valid filter value for filtering

    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
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
    saveloc = f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/testevalfinal1/fullimg/{datasetname}/{modelname}/iou_{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}"
    # boxsaveloc = f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/{datasetname}/{modelname}"
    if filter:
        with open(
            f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/bestfilterthresholds{'_valid' if valid else ''}/{modelname}.txt",
            "r",
        ) as f:
            filtering = float(f.read())
    if tableareaonly and not filter:
        saveloc = f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/testevalfinal1/fullimg/{datasetname}/{modelname}/tableareaonly/iou_{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}"
    elif filter and not tableareaonly:
        saveloc = f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/testevalfinal1/fullimg/{datasetname}/{modelname}/filtering_{filtering}{'_valid' if valid else ''}_iou{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}"
    elif filter and tableareaonly:
        saveloc = f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/testevalfinal1/fullimg/{datasetname}/{modelname}/tableareaonly/filtering_{filtering}{'_valid' if valid else ''}_iou{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}"
    os.makedirs(saveloc, exist_ok=True)

    # *** initializing variables ***
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
    # *** initializing variables ***

    for n, folder in tqdm(enumerate(glob.glob(f"{targetloc}/*"))):
        impath = f"{folder}/{folder.split('/')[-1]}.jpg"
        imname = folder.split("/")[-1]
        if tablerelative:
            fullimagegroundbox = reversetablerelativebboxes_outer(folder)
        else:
            fullimagegroundbox = torch.load(
                glob.glob(f"{folder}/{folder.split('/')[-1]}.pt")[0]
            )
        img = (read_image(impath) / 255).to(device)
        output = model([img])
        output = {k: v.detach().cpu() for k, v in output[0].items()}
        # print(output['boxes'], output['boxes'][output['scores']>0.8])
        if filter:
            output["boxes"] = output["boxes"][output["scores"] > filtering]
        if tableareaonly:
            output["boxes"] = tableareabboxes(output["boxes"], folder)
        fullimagepredbox = output["boxes"]

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
    totaloverlapdf = pandas.DataFrame(
        {"f1": overlapf1, "prec": overlaprec, "recall": overlaprec}, index=["overlap"]
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


def inference_tablecutout(
    datapath: str,  # f"{Path(__file__).parent.absolute()}/../../data/BonnData/test"
    modelpath: str,  # f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/run3_BonnData_cell_aug_loadrun_GloSAT_cell_aug_e250_es_e250_es.pt",
    datasetname: str = "BonnData",
    iou_thresholds: List[float] = None,  # [0.5, 0.6, 0.7, 0.8, 0.9]
    filtering=False,
    saveboxes=False,
):
    """Inference on cut out table images.

    Args:
        datapath: folder with preprocessed test data (images and BBoxes)
        modelpath: path to model checkpoint
        datasetname: name of the dataset (for save locations)
        iou_thresholds: iou threshold list
        filtering: wether to filter the results
        saveboxes: wether to save result of inference

    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        **{"box_detections_per_img": 200},
    )
    model.load_state_dict(torch.load(modelpath))
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        # model.eval()
    else:
        print("Cuda not available")
        return
    model.eval()
    modelname = modelpath.split(os.sep)[-1]
    saveloc = f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/testevalfinal1/tableareacutout/{datasetname}/{modelname}"
    boxsaveloc = f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/tableareacutout/{datasetname}/{modelname}"
    if filtering:
        saveloc = f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/testevalfinal1/tableareacutout/{datasetname}/{modelname}/filtering"
    os.makedirs(saveloc, exist_ok=True)
    if saveboxes:
        os.makedirs(boxsaveloc, exist_ok=True)

    ioulist = []
    f1list = []
    wf1list = []
    tpsum = torch.zeros(len(iou_thresholds))
    fpsum = torch.zeros(len(iou_thresholds))
    fnsum = torch.zeros(len(iou_thresholds))
    ioudf = pandas.DataFrame(
        columns=["img", "mean pred iou", "mean tar iou", "wf1", "prednum"]
    )
    # .............................
    tpsum_iodt = torch.zeros(len(iou_thresholds))
    fpsum_iodt = torch.zeros(len(iou_thresholds))
    fnsum_iodt = torch.zeros(len(iou_thresholds))
    iodtlist = []
    f1list_iodt = []
    wf1list_iodt = []
    iodtdf = pandas.DataFrame(
        columns=["img", "mean pred iodt", "mean tar iodt", "wf1", "prednum"]
    )
    for n, folder in enumerate(glob.glob(f"{datapath}/*")):
        for tableimg in glob.glob(f"{folder}/*table_*pt"):
            num = tableimg.split(".")[-2].split("_")[-1]
            img = (torch.load(tableimg) / 255).to(device)
            target = torch.load(f"{folder}/{folder.split('/')[-1]}_cell_{num}.pt")
            output = model([img])
            output = {k: v.detach().cpu() for k, v in output[0].items()}
            if filtering:
                output["boxes"] = output["boxes"][output["scores"] > 0.7]

            if saveboxes:
                os.makedirs(f"{boxsaveloc}/{folder.split('/')[-1]}", exist_ok=True)
                torch.save(
                    output["boxes"],
                    f"{boxsaveloc}/{folder.split('/')[-1]}/{folder.split('/')[-1]}_{num}.pt",
                )

            # ..................IoU
            prediou, targetiou, tp, fp, fn = calcstats_iou(
                predbox=output["boxes"],
                targetbox=target,
                iou_thresholds=iou_thresholds,
                imname=folder.split("/")[-1] + "_" + num,
            )
            prec, rec, f1, wf1 = calcmetric(tp, fp, fn, iou_thresholds=iou_thresholds)
            tpsum += tp
            fpsum += fp
            fnsum += fn
            ioulist.append(prediou)
            f1list.append(f1)
            wf1list.append(wf1)
            ioumetrics = {
                "img": folder.split("/")[-1],
                "mean pred iou": torch.mean(prediou).item(),
                "mean tar iou": torch.mean(targetiou).item(),
                "wf1": wf1.item(),
                "prednum": output["boxes"].shape[0],
            }
            ioumetrics.update(
                {
                    f"prec@{iou_thresholds[i]}": prec[i].item()
                    for i in range(len(iou_thresholds))
                }
            )
            ioumetrics.update(
                {
                    f"recall@{iou_thresholds[i]}": rec[i].item()
                    for i in range(len(iou_thresholds))
                }
            )
            ioumetrics.update(
                {
                    f"f1@{iou_thresholds[i]}": f1[i].item()
                    for i in range(len(iou_thresholds))
                }
            )
            ioumetrics.update(
                {
                    f"tp@{iou_thresholds[i]}": tp[i].item()
                    for i in range(len(iou_thresholds))
                }
            )
            ioumetrics.update(
                {
                    f"fp@{iou_thresholds[i]}": fp[i].item()
                    for i in range(len(iou_thresholds))
                }
            )
            ioumetrics.update(
                {
                    f"fn@{iou_thresholds[i]}": fn[i].item()
                    for i in range(len(iou_thresholds))
                }
            )
            ioudf = pandas.concat(
                [ioudf, pandas.DataFrame(ioumetrics, index=[f"{n}.{num}"])]
            )
            # ...........................
            # ....................IoDt
            prediodt, targetiodt, tp_iodt, fp_iodt, fn_iodt = calcstats_iodt(
                predbox=output["boxes"],
                targetbox=target,
                iou_thresholds=iou_thresholds,
                imname=folder.split("/")[-1] + "_" + num,
            )
            prec_iodt, rec_iodt, f1_iodt, wf1_iodt = calcmetric(
                tp_iodt, fp_iodt, fn_iodt, iou_thresholds=iou_thresholds
            )
            tpsum_iodt += tp_iodt
            fpsum_iodt += fp_iodt
            fnsum_iodt += fn_iodt
            iodtlist.append(prediodt)
            f1list_iodt.append(f1_iodt)
            wf1list_iodt.append(wf1_iodt)
            iodtmetrics = {
                "img": folder.split("/")[-1],
                "mean pred iodt": torch.mean(prediodt).item(),
                "mean tar iodt": torch.mean(targetiodt).item(),
                "wf1": wf1_iodt.item(),
                "prednum": output["boxes"].shape[0],
            }
            iodtmetrics.update(
                {
                    f"prec@{iou_thresholds[i]}": prec_iodt[i].item()
                    for i in range(len(iou_thresholds))
                }
            )
            iodtmetrics.update(
                {
                    f"recall@{iou_thresholds[i]}": rec_iodt[i].item()
                    for i in range(len(iou_thresholds))
                }
            )
            iodtmetrics.update(
                {
                    f"f1@{iou_thresholds[i]}": f1_iodt[i].item()
                    for i in range(len(iou_thresholds))
                }
            )
            iodtmetrics.update(
                {
                    f"tp@{iou_thresholds[i]}": tp_iodt[i].item()
                    for i in range(len(iou_thresholds))
                }
            )
            iodtmetrics.update(
                {
                    f"fp@{iou_thresholds[i]}": fp_iodt[i].item()
                    for i in range(len(iou_thresholds))
                }
            )
            iodtmetrics.update(
                {
                    f"fn@{iou_thresholds[i]}": fn_iodt[i].item()
                    for i in range(len(iou_thresholds))
                }
            )
            iodtdf = pandas.concat(
                [iodtdf, pandas.DataFrame(iodtmetrics, index=[f"{n}.{num}"])]
            )

    conclusiondf = pandas.DataFrame(columns=["wf1"])
    fulliodtdf = get_dataframe(
        fnsum=fnsum_iodt,
        fpsum=fpsum_iodt,
        tpsum=tpsum_iodt,
        iou_thresholds=iou_thresholds,
    )
    fullioudf = get_dataframe(
        fnsum=fnsum, fpsum=fpsum, tpsum=tpsum, iou_thresholds=iou_thresholds
    )
    conclusiondf = pandas.concat(
        [
            conclusiondf,
            pandas.DataFrame(fullioudf, index=["IoU"]),
            pandas.DataFrame(fulliodtdf, index=["IoDt"]),
        ]
    )
    ioudf.to_csv(f"{saveloc}/iou.csv")
    iodtdf.to_csv(f"{saveloc}/iodt.csv")
    conclusiondf.to_csv(f"{saveloc}/overview.csv")
    print("done")


if __name__ == "__main__":
    """
    inference_fullimg(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/testseveralcalls_no_valid_random_init_e_250_end.pt"
    )
    for cat in glob.glob(
            f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/*"
    ):
        print(cat)
        inference_fullimg(
            targetloc=cat, datasetname=f"Tablesinthewild/{cat.split('/')[-1]}",
            modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/testseveralcalls_no_valid_random_init_e_250_es.pt"
        )
        inference_fullimg(
            targetloc=cat, datasetname=f"Tablesinthewild/{cat.split('/')[-1]}",
            modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/testseveralcalls_no_valid_random_init_e_250_es.pt",
            tableareaonly=True
        )

    for cat in glob.glob(
            f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/*"
    ):
        print(cat)
        inference_fullimg(
            targetloc=cat, datasetname=f"Tablesinthewild/{cat.split('/')[-1]}", modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/testseveralcalls_valid_random_init_e_250_es.pt"
        )
        inference_fullimg(
            targetloc=cat, datasetname=f"Tablesinthewild/{cat.split('/')[-1]}",
            modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/testseveralcalls_valid_random_init_e_250_es.pt", tableareaonly=True
        )
    inference_fullimg(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test",
        datasetname="BonnData",
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/BonnDataFullImage_random_BonnData_fullimage_e250_random_init_es.pt",
        tablerelative=True,
        tableareaonly=True,
    )
    inference_fullimg(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
        datasetname="GloSat",
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/GloSatFullImage_random_GloSat_fullimage_e250_random_init_es.pt",
        tablerelative=True,
        tableareaonly=True,
    )
    inference_fullimg(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test",
        datasetname="BonnData",
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/BonnDataFullImage_random_BonnData_fullimage_e250_random_init_es.pt",
        tablerelative=True,
        tableareaonly=False,
    )
    inference_fullimg(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
        datasetname="GloSat",
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/GloSatFullImage_random_GloSat_fullimage_e250_random_init_es.pt",
        tablerelative=True,
        tableareaonly=False,
    )

    """
    inference_fullimg(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test",
        datasetname="BonnData",
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/BonnDataFullImage1_BonnData_fullimage_e250_es.pt",
        tablerelative=True,
        tableareaonly=True,
        filter=True,
    )
    inference_fullimg(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
        datasetname="GloSat",
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/GloSatFullImage1_GloSat_fullimage_e250_es.pt",
        tablerelative=True,
        tableareaonly=True,
        filter=True,
    )
    inference_fullimg(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test",
        datasetname="BonnData",
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/BonnDataFullImage1_BonnData_fullimage_e250_es.pt",
        tablerelative=True,
        tableareaonly=False,
        filter=True,
    )
    inference_fullimg(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
        datasetname="GloSat",
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/GloSatFullImage1_GloSat_fullimage_e250_es.pt",
        tablerelative=True,
        tableareaonly=False,
        filter=True,
    )

    """
    """
    for cat in glob.glob(
        f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/*"
    ):
        print(cat)
        inference_fullimg(
            targetloc=cat,
            datasetname=f"Tablesinthewild/{cat.split('/')[-1]}/tablerelative",
            modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/testseveralcalls_4_with_valid_split_Tablesinthewild_fullimage_e50_es.pt",
            tablerelative=True,
            tableareaonly=True,
        )
        inference_fullimg(
            targetloc=cat,
            datasetname=f"Tablesinthewild/{cat.split('/')[-1]}/tablerelative",
            modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/testseveralcalls_5_without_valid_split_Tablesinthewild_fullimage_e50_end.pt",
            tablerelative=True,
            tableareaonly=True,
        )
    for cat in glob.glob(
        f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/*"
    ):
        print(cat)
        inference_fullimg(
            targetloc=cat,
            datasetname=f"Tablesinthewild/{cat.split('/')[-1]}",
            modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/testseveralcalls_4_with_valid_split_Tablesinthewild_fullimage_e50_es.pt",
            tableareaonly=True,
        )
        inference_fullimg(
            targetloc=cat,
            datasetname=f"Tablesinthewild/{cat.split('/')[-1]}",
            modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/testseveralcalls_5_without_valid_split_Tablesinthewild_fullimage_e50_end.pt",
            tableareaonly=True,
        )
    # inference_fullimg(
    #    modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/testseveralcalls_5_without_valid_split_Tablesinthewild_fullimage_e50_end.pt",
    #    targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/test1",
    #    datasetname="Tablesinthewild_oldxml",
    # )
    inference_fullimg()
    inference_fullimg(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/test5_with_valid_split_Tablesinthewild_fullimage_e50_es.pt"
    )
    inference_fullimg(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/testseveralcalls_4_with_valid_split_Tablesinthewild_fullimage_e50_es.pt"
    )
    inference_fullimg(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/testseveralcalls_4_with_valid_split_Tablesinthewild_fullimage_e50_end.pt"
    )
    inference_fullimg(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/testseveralcalls_5_without_valid_split_Tablesinthewild_fullimage_e50_es.pt"
    )
    inference_fullimg(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/testseveralcalls_5_without_valid_split_Tablesinthewild_fullimage_e50_end.pt"
    )
    inference_fullimg(iou_thresholds=[0.9])
    for cat in glob.glob(
        f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/*"
    ):
        print(cat)
        inference_fullimg(
            targetloc=cat, datasetname=f"Tablesinthewild/{cat.split('/')[-1]}"
        )
    for cat in glob.glob(
        f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/*"
    ):
        print(cat)
        inference_fullimg(
            targetloc=cat,
            datasetname=f"Tablesinthewild/{cat.split('/')[-1]}",
            modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/testseveralcalls_4_with_valid_split_Tablesinthewild_fullimage_e50_es.pt",
        )
        inference_fullimg(
            targetloc=cat,
            datasetname=f"Tablesinthewild/{cat.split('/')[-1]}",
            modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/testseveralcalls_5_without_valid_split_Tablesinthewild_fullimage_e50_end.pt",
        )
    for cat in glob.glob(
        f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/*"
    ):
        print(cat)
        inference_fullimg(
            targetloc=cat,
            datasetname=f"Tablesinthewild/{cat.split('/')[-1]}",
            modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/test5_with_valid_split_Tablesinthewild_fullimage_e50_es.pt",
        )
    inference_fullimg(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test",
        datasetname="BonnData",
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/BonnDataFullImage1_BonnData_fullimage_e250_es.pt",
        tablerelative=True,
        tableareaonly=True,
    )
    inference_fullimg(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
        datasetname="GloSat",
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/GloSatFullImage1_GloSat_fullimage_e250_es.pt",
        tablerelative=True,
        tableareaonly=True,
    )
    inference_fullimg(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test",
        datasetname="BonnData",
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/BonnDataFullImage1_BonnData_fullimage_e250_es.pt",
        tablerelative=True,
        tableareaonly=False,
    )
    inference_fullimg(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
        datasetname="GloSat",
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/GloSatFullImage1_GloSat_fullimage_e250_es.pt",
        tablerelative=True,
        tableareaonly=False,
    )
    inference_fullimg(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test",
        datasetname="BonnData",
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_es.pt",
        tablerelative=True,
        tableareaonly=True,
    )
    inference_fullimg(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test",
        datasetname="BonnData",
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_end.pt",
        tablerelative=True,
        tableareaonly=True,
    )
    inference_fullimg(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test",
        datasetname="BonnData",
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_es.pt",
        tablerelative=True,
        tableareaonly=False,
    )
    inference_fullimg(
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test",
        datasetname="BonnData",
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_end.pt",
        tablerelative=True,
        tableareaonly=False,
    )
    """

    # inference_fullimg(targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/simple", datasetname='simple')
    inference_tablecutout()
    # inference_tablecutout(filtering=True)
    inference_tablecutout(
        datapath=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn"
        f"/run_GloSAT_cell_aug_e250_es.pt",
        datasetname="GloSat",
    )
    inference_tablecutout(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/run_BonnData_cell_e250_es.pt"
    )
    inference_tablecutout(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/run3_BonnData_cell_loadrun_GloSAT_cell_e250_es_e250_es.pt"
    )
    inference_tablecutout(
        datapath=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn"
        f"/run_GloSAT_cell_e250_es.pt",
        datasetname="GloSat",
    )
    # inference_tablecutout(datapath=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
    #                      modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn"
    #                                f"/run_GloSAT_cell_aug_e250_es.pt",
    #                      datasetname="GloSat", filtering=True)
    # inference_tablecutout(
    #    modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/run_BonnData_cell_e250_es.pt", filtering=True)
    # inference_tablecutout(
    #    modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/run3_BonnData_cell_loadrun_GloSAT_cell_e250_es_e250_es.pt", filtering=True)
    # inference_tablecutout(datapath=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
    #                      modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn"
    #                                f"/run_GloSAT_cell_e250_es.pt",
    #                      datasetname="GloSat", filtering=True)
    """
