import glob
import json
import os
from pathlib import Path
from typing import Literal

import pandas
import torch
from PIL import Image
from torchaudio.functional import filtering
from torchvision.io import read_image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm
from typing_extensions import List

from src.historicdocumentprocessing.fasterrcnn_eval import tableareabboxes
from src.historicdocumentprocessing.kosmos_eval import (
    calcmetric,
    calcmetric_overlap,
    calcstats_IoDT,
    calcstats_IoU,
    calcstats_overlap,
    extractboxes,
    get_dataframe,
    reversetablerelativebboxes_outer,
)
from src.historicdocumentprocessing.util.glosat_paper_postprocessing_method import (
    reconstruct_table,
)
from src.historicdocumentprocessing.util.tablesutil import (
    clustertables,
    clustertablesseperately,
    getsurroundingtable,
    remove_invalid_bbox,
)


def reconstruct_bboxes(rows: list, colls: list, tablecoords: list) -> torch.Tensor:
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
    # print(rows, colls)
    return torch.tensor(boxes)


def postprocess_kosmos(
    targetloc: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData/test",
    predloc: str = f"{Path(__file__).parent.absolute()}/../../results/kosmos25/BonnData"
    f"/Tabellen/test",
    datasetname: str = "BonnData",
):
    saveloc = f"{Path(__file__).parent.absolute()}/../../results/kosmos25/postprocessed/fullimg/{datasetname}"
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
            # print(targets[0])
            # print("here")
            postprocess(fullimagepredbox, imname, saveloc)


def postprocess(
    fullimagepredbox: torch.Tensor,
    imname: str,
    saveloc: str,
    includeoutlier: bool = False,
    includeoutliers_cellsearch:bool = False
):
    # for epsprefactor in [tuple([3.0,1.5]), tuple([4.0,1.5])]:
    epsprefactor = tuple([3.0, 1.5])
    clusteredtables = clustertablesseperately(
        fullimagepredbox, epsprefactor=epsprefactor, includeoutlier=includeoutlier
    )
    # print(len(clusteredtables))
    # if tableavail and clusteredtables:
    #    tablenum = len(glob.glob(f"{targets}/*cell*"))
    #    fullimagegroundbox = reversetablerelativebboxes_outer(targets)
    #    if tablenum != len(clusteredtables):
    #        print("imname: ", imname, epsprefactor, tablenum, len(clusteredtables))
    boxes = []
    saveloc = f"{saveloc}/eps_{epsprefactor[0]}_{epsprefactor[1]}"
    if not includeoutlier:
        saveloc = f"{saveloc}/withoutoutlier"
    else:
        saveloc = f"{saveloc}/withoutlier"
    saveloc = f"{f'{saveloc}/includeoutliers_glosat' if includeoutliers_cellsearch else f'{saveloc}/excludeoutliers_glosat'}"
    saveloc = f"{saveloc}/{imname}"
    print(saveloc)
    # img = read_image(glob.glob(f"{targets}/*jpg")[0])
    for i, t in enumerate(clusteredtables):
        tablecoord = getsurroundingtable(t).tolist()
        rows, colls = reconstruct_table(t.tolist(), tablecoord, eps=4, include_outliers=includeoutliers_cellsearch)
        # print(imname, rows, colls)
        tbboxes = reconstruct_bboxes(rows, colls, tablecoord)
        boxes.append(tbboxes)
        # img = read_image(glob.glob(f"{targets}/*jpg")[0])
        # colors = ["green" for i in range(t.shape[0])]
        # res = draw_bounding_boxes(image=img, boxes=t, colors=colors)
        # res = Image.fromarray(res.permute(1, 2, 0).numpy())
        #                # print(f"{savepath}/{identifier}.jpg")
        # res.save(f"{Path(__file__).parent.absolute()}/../../images/test/test_complete_{imname}_{i}.jpg")
    # print(boxes)
    # print(len(boxes))
    if len(boxes) > 0:
        # print(saveloc)
        bboxes = torch.vstack(boxes)
        # print(saveloc)
        os.makedirs(saveloc, exist_ok=True)
        torch.save(bboxes, f"{saveloc}/{imname}.pt")
        # print(f"{saveloc}/{imname}.pt")
    else:
        # print(imname)
        # print(clusteredtables)
        # print(boxes)
        # print(fullimagepredbox)
        print(f"no valid predictions for image {imname}")
    # print(boxes)
    # colors = ["green" for i in range(boxes.shape[0])]
    # res = draw_bounding_boxes(image=img, boxes=boxes, colors=colors)
    # res = Image.fromarray(res.permute(1, 2, 0).numpy())
    #                # print(f"{savepath}/{identifier}.jpg")
    # res.save(f"{Path(__file__).parent.absolute()}/../../images/test/test_complete_{imname}.jpg")

    #            if t.shape[0]>1:
    #                img = read_image(glob.glob(f"{targets}/*jpg")[0])
    #                colors = ["green" for i in range(t.shape[0])]
    #                res = draw_bounding_boxes(image=img, boxes=t, colors=colors)
    #                res = Image.fromarray(res.permute(1, 2, 0).numpy())
    #                # print(f"{savepath}/{identifier}.jpg")
    #                res.save(f"{Path(__file__).parent.absolute()}/../../images/test/test_kosmos1_{imname}_{i}.jpg")
    # else:
    #    fullimagegroundbox = torch.load(
    #        glob.glob(f"{targetloc}/{preds.split('/')[-1]}/{preds.split('/')[-1]}.pt")[0])


def postprocess_rcnn(modelpath=None, targetloc=None, datasetname=None):
    saveloc = f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/postprocessed/fullimg/{datasetname}"
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
    for n, folder in tqdm(enumerate(glob.glob(f"{targetloc}/*"))):
        # print(folder)
        impath = f"{folder}/{folder.split('/')[-1]}.jpg"
        imname = folder.split("/")[-1]
        img = (read_image(impath) / 255).to(device)
        output = model([img])
        output = {k: v.detach().cpu() for k, v in output[0].items()}
        # print(output['boxes'], output['boxes'][output['scores']>0.8])
        if filtering:
            output["boxes"] = output["boxes"][output["scores"] > 0.8]
        fullimagepredbox = remove_invalid_bbox(output["boxes"])
        postprocess(fullimagepredbox, imname, saveloc)


def postprocess_eval(
    datasetname: str,
    modeltype: Literal["kosmos25", "fasterrcnn"] = "kosmos25",
    targetloc: str = None,
    modelpath: str = None,
    tablerelative=True,
    iou_thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9],
    tableareaonly=False,
    includeoutliers_cellsearch=False
):
    predloc = f"{Path(__file__).parent.absolute()}/../../results/{modeltype}/postprocessed/fullimg/{datasetname}"
    saveloc = f"{Path(__file__).parent.absolute()}/../../results/{modeltype}/testevalfinal1/fullimg/{datasetname}"
    if modelpath:
        modelname = modelpath.split(os.sep)[-1]
        predloc = f"{predloc}/{modelname}"
        saveloc = f"{saveloc}/{modelname}/iou_{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}/postprocessed/eps_3.0_1.5/withoutoutlier"
    else:
        saveloc = f"{saveloc}/iou_{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}/postprocessed/eps_3.0_1.5/withoutoutlier"
    saveloc = f"{f'{saveloc}/includeoutliers_glosat' if includeoutliers_cellsearch else f'{saveloc}/excludeoutliers_glosat'}"
    predloc = f"{predloc}/eps_3.0_1.5/withoutoutlier"
    predloc = f"{f'{predloc}' if includeoutliers_cellsearch else f'{predloc}/excludeoutliers_glosat'}"
    print(saveloc)
    print(predloc)
    # saveloc = f"{Path(__file__).parent.absolute()}/../../results/{modeltype}/testevalfinal1/fullimg/{datasetname}/{modelname}/iou_{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}/postprocessed"
    ### initializing variables ###
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
    tpsum_IoDT = torch.zeros(len(iou_thresholds))
    fpsum_IoDT = torch.zeros(len(iou_thresholds))
    fnsum_IoDT = torch.zeros(len(iou_thresholds))
    tpsum_IoDT_predonly = torch.zeros(len(iou_thresholds))
    fpsum_IoDT_predonly = torch.zeros(len(iou_thresholds))
    fnsum_IoDT_predonly = torch.zeros(len(iou_thresholds))
    predcount = 0
    overlapdf = pandas.DataFrame(columns=["img", "prednum"])
    fullimagedf = pandas.DataFrame(
        columns=["img", "mean pred iou", "mean tar iou", "wf1", "prednum"]
    )
    iodtdf = pandas.DataFrame(
        columns=["img", "mean pred iod", "mean tar iod", "wf1", "prednum"]
    )
    ### initializing variables ###
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
        prediod, tariod, IoDT_tp, IoDT_fp, IoDT_fn = calcstats_IoDT(
            predbox=fullimagepredbox,
            targetbox=fullimagegroundbox,
            imname=imname,
            iou_thresholds=iou_thresholds,
        )
        # print(IoDT_tp, IoDT_fp, IoDT_fn)
        IoDT_prec, IoDT_rec, IoDT_f1, IoDT_wf1 = calcmetric(
            IoDT_tp, IoDT_fp, IoDT_fn, iou_thresholds=iou_thresholds
        )
        # print(IoDT_prec, IoDT_rec, IoDT_f1, IoDT_wf1)
        tpsum_IoDT += IoDT_tp
        fpsum_IoDT += IoDT_fp
        fnsum_IoDT += IoDT_fn
        IoDTmetrics = {
            "img": imname,
            "mean pred iod": torch.mean(prediod).item(),
            "mean tar iod": torch.mean(tariod).item(),
            "wf1": IoDT_wf1.item(),
            "prednum": fullimagepredbox.shape[0],
        }
        IoDTmetrics.update(
            {
                f"prec@{iou_thresholds[i]}": IoDT_prec[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        IoDTmetrics.update(
            {
                f"recall@{iou_thresholds[i]}": IoDT_rec[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        IoDTmetrics.update(
            {
                f"f1@{iou_thresholds[i]}": IoDT_f1[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        IoDTmetrics.update(
            {
                f"tp@{iou_thresholds[i]}": IoDT_tp[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        IoDTmetrics.update(
            {
                f"fp@{iou_thresholds[i]}": IoDT_fp[i].item()
                for i in range(len(iou_thresholds))
            }
        )
        IoDTmetrics.update(
            {
                f"fn@{iou_thresholds[i]}": IoDT_fn[i].item()
                for i in range(len(iou_thresholds))
            }
        )

        iodtdf = pandas.concat([iodtdf, pandas.DataFrame(IoDTmetrics, index=[n])])

        # .................................
        # fullimagemetrics with iou
        # .................................
        fullprediou, fulltargetiou, fulltp, fullfp, fullfn = calcstats_IoU(
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
            f"prec": fullprec_overlap.item(),
            f"recall": fullrec_overlap.item(),
            f"f1": fullf1_overlap.item(),
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
            tpsum_IoDT_predonly += IoDT_tp
            fpsum_IoDT_predonly += IoDT_fp
            fnsum_IoDT_predonly += IoDT_fn
            predcount += 1

            # print("here",fullimagepredbox)
        # print(fullfp, fullimagemetrics)
        fullimagedf = pandas.concat(
            [fullimagedf, pandas.DataFrame(fullimagemetrics, index=[n])]
        )
        # print(fullimagedf.loc[n])

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
    # print(totalfullmetrics)
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
            f"Number of evaluated files": overlapdf.shape[0],
            f"Evaluated files without predictions:": overlapdf.shape[0] - predcount,
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
        fnsum=fnsum_IoDT,
        fpsum=fpsum_IoDT,
        tpsum=tpsum_IoDT,
        nopredcount=iodtdf.shape[0] - predcount,
        imnum=iodtdf.shape[0],
        iou_thresholds=iou_thresholds,
    )
    predonlyiodt = get_dataframe(
        fnsum=fnsum_IoDT_predonly,
        fpsum=fpsum_IoDT_predonly,
        tpsum=tpsum_IoDT_predonly,
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

    # print(conclusiondf)
    # save results
    os.makedirs(saveloc, exist_ok=True)
    # print(fullimagedf.loc[50])
    # print(saveloc)
    overlapdf.to_csv(f"{saveloc}/fullimageoverlapeval.csv")
    fullimagedf.to_csv(f"{saveloc}/fullimageiou.csv")
    iodtdf.to_csv(f"{saveloc}/fullimageiodt.csv")
    conclusiondf.to_csv(f"{saveloc}/overview.csv")


if __name__ == "__main__":
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
    postprocess_eval(
        datasetname="BonnData",
        modeltype="kosmos25",
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test",
    )
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
    postprocess_eval(
        datasetname="GloSat",
        modeltype="kosmos25",
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
    )
    postprocess_eval(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/GloSatFullImage1_GloSat_fullimage_e250_es.pt",
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
        datasetname="GloSat",
        modeltype="fasterrcnn",
    )

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

    """
    """
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
