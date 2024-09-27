import glob
import os
from pathlib import Path
from typing import List

import pandas
import torch
from PIL import Image
from lightning.fabric.utilities import move_data_to_device
from tqdm import tqdm
from  transformers import AutoModelForObjectDetection, DetrImageProcessor, TableTransformerForObjectDetection

from src.historicdocumentprocessing.fasterrcnn_eval import tableareabboxes
from src.historicdocumentprocessing.kosmos_eval import reversetablerelativebboxes_outer, calcstats_IoDT, calcmetric, \
    calcstats_IoU, calcstats_overlap, calcmetric_overlap, get_dataframe
from src.historicdocumentprocessing.tabletransformer_dataset import CustomDataset
from src.historicdocumentprocessing.util.tablesutil import getcells


def inference(modelpath:str=None, targetloc=None, iou_thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9], tableareaonly=True,
              filtering:bool = False, valid:bool = True, datasetname:str="BonnData", celleval:bool=False):
    modelname="base_table-transformer-structure-recognition"
    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
    if modelpath:
        model.load_state_dict(torch.load(modelpath))
        modelname = modelpath.split('/')[-1]
        print("loaded_model:", modelname)
    assert model.config.id2label[1]=="table column"
    assert model.config.id2label[2]=="table row"
    #image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    #image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
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
                'r') as f:
            filterthreshold = float(f.read())
    saveloc = f"{Path(__file__).parent.absolute()}/../../results/tabletransformer/testevalfinal1/fullimg/{datasetname}/{modelname}/iou_{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}"
    # boxsaveloc = f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/{datasetname}/{modelname}"
    if tableareaonly and not filtering:
        saveloc = f"{Path(__file__).parent.absolute()}/../../results/tabletransformer/testevalfinal1/fullimg/{datasetname}/{modelname}/tableareaonly/iou_{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}"
    elif filtering and not tableareaonly:
        saveloc = f"{Path(__file__).parent.absolute()}/../../results/tabletransformer/testevalfinal1/fullimg/{datasetname}/{modelname}/filtering_{filterthreshold}_iou{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}"
    elif filtering and tableareaonly:
        saveloc = f"{Path(__file__).parent.absolute()}/../../results/tabletransformer/testevalfinal1/fullimg/{datasetname}/{modelname}/tableareaonly/filtering_{filterthreshold}_iou{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}"
    saveloc=f"{saveloc}{'/cells' if celleval else ''}"
    os.makedirs(saveloc, exist_ok=True)

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

    dataset = CustomDataset(
        targetloc,
        "fullimage",
        transforms=None,
    )
    for i in tqdm(range(len(dataset))):
        img, fullimagegroundbox, labels = dataset.getimgtarget(i, addtables=False)
        if celleval:
            fullimagegroundbox, labels = dataset.getcells(i, addtables=False)
        encoding = move_data_to_device(dataset.ImageProcessor(img, return_tensors="pt"), device=device)
        folder = dataset.getfolder(i)
        imname = folder.split("/")[-1]

        with torch.no_grad():
            results = model(**encoding)
        width, height = img.size
        if filtering:
            output = dataset.ImageProcessor.post_process_object_detection(results, threshold=filterthreshold, target_sizes=[(height, width)])[0]
        else:
            output = dataset.ImageProcessor.post_process_object_detection(results, target_sizes=[(height, width)])[0]
        #labels = torch.Tensor([str(model.config.id2label[label]) for label in output["labels"].tolist()])
        if celleval:
            boxes = getcells(rows=output["boxes"][output["labels"]==2], cols=output["boxes"][output["labels"]==1])
        else:
            boxes = torch.vstack([output["boxes"][output["labels"]==2], output["boxes"][output["labels"]==1]])
        print(boxes.shape)
        if tableareaonly:
            boxes = tableareabboxes(boxes, folder)
        fullimagepredbox = boxes.to(device="cpu")
        print(boxes.shape)

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

        iodtdf = pandas.concat([iodtdf, pandas.DataFrame(IoDTmetrics, index=[i])])

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
            [overlapdf, pandas.DataFrame(overlapmetric, index=[i])]
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
            [fullimagedf, pandas.DataFrame(fullimagemetrics, index=[i])]
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
    # os.makedirs(saveloc, exist_ok=True)
    # print(fullimagedf.loc[50])
    # print(saveloc)
    overlapdf.to_csv(f"{saveloc}/fullimageoverlapeval.csv")
    fullimagedf.to_csv(f"{saveloc}/fullimageiou.csv")
    iodtdf.to_csv(f"{saveloc}/fullimageiodt.csv")
    conclusiondf.to_csv(f"{saveloc}/overview.csv")

if __name__=='__main__':
    inference(modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/tabletransformer/tabletransformer_v0_new_BonnDataFullImage_tabletransformer_v0_bbox_xywh_BonnData_fullimage_e250_valid_end.pt",
              targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test")
    inference(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/tabletransformer/tabletransformer_v0_new_BonnDataFullImage_tabletransformer_estest_BonnData_fullimage_e250_valid_es.pt",
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test")
    inference(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/tabletransformer/tabletransformer_v0_new_BonnDataFullImage_tabletransformer_estest_BonnData_fullimage_e250_valid_end.pt",
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test")
    inference(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/tabletransformer/tabletransformer_v0_new_GloSatFullImage_tabletransformer_newenv_fixed_GloSat_fullimage_e250_valid_es.pt",
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test", datasetname="GloSat")
    inference(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/tabletransformer/tabletransformer_v0_new_GloSatFullImage_tabletransformer_newenv_fixed_GloSat_fullimage_e250_valid_end.pt",
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test", datasetname="GloSat")
    inference(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/tabletransformer/tabletransformer_v0_new_BonnDataFullImage_tabletransformer_v0_bbox_xywh_BonnData_fullimage_e250_valid_end.pt",
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test", celleval=True)
    inference(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/tabletransformer/tabletransformer_v0_new_BonnDataFullImage_tabletransformer_estest_BonnData_fullimage_e250_valid_es.pt",
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test", celleval=True)
    inference(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/tabletransformer/tabletransformer_v0_new_BonnDataFullImage_tabletransformer_estest_BonnData_fullimage_e250_valid_end.pt",
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test", celleval=True)
    inference(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/tabletransformer/tabletransformer_v0_new_GloSatFullImage_tabletransformer_newenv_fixed_GloSat_fullimage_e250_valid_es.pt",
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test", datasetname="GloSat", celleval=True)
    inference(
        modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/tabletransformer/tabletransformer_v0_new_GloSatFullImage_tabletransformer_newenv_fixed_GloSat_fullimage_e250_valid_end.pt",
        targetloc=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test", datasetname="GloSat", celleval=True)