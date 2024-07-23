import glob
import os
from pathlib import Path
from typing import List

import pandas
import torch

from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
from torchvision.io import read_image

from src.historicdocumentprocessing.kosmos_eval import calcstats, calcmetric, reversetablerelativebboxes_outer, calcstats_overlap, calcmetric_overlap
from src.historicdocumentprocessing.kosmos_eval import calcstats_IoDT, get_dataframe

from tqdm import tqdm


def inference_fullimg(targetloc: str = f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/test",
                      modelpath: str = f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn"
                                       f"/test4_Tablesinthewild_fullimage_e50_end.pt",
                      datasetname: str = "Tablesinthewild",
                      iou_thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9], filtering=False,
                      tablerelative: bool = False, tableareaonly: bool = False):
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, **{"box_detections_per_img": 200}
    )
    model.load_state_dict(
        torch.load(
            modelpath
        )
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        model.eval()
    else:
        print("Cuda not available")
        return
    modelname = modelpath.split(os.sep)[-1]
    saveloc = f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/testeval/fullimg/{datasetname}/{modelname}/iou_{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}"
    if tableareaonly and not filtering:
        saveloc = f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/testeval/fullimg/{datasetname}/{modelname}/tableareaonly/iou_{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}"
    #boxsaveloc = f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/{datasetname}"
    elif filtering and not tableareaonly:
        saveloc = f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/testeval/fullimg/{datasetname}/{modelname}/filtering_iou{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}"
    elif filtering and tableareaonly:
        saveloc = f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/testeval/fullimg/{datasetname}/{modelname}/tableareaonly/filtering_iou{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}"
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
    fullimagedf = pandas.DataFrame(columns=["img", "mean pred iou", "mean tar iou", "wf1", "prednum"])
    iodtdf = pandas.DataFrame(columns=["img", "mean pred iod", "mean tar iod", "wf1", "prednum"])
    ### initializing variables ###

    for n, folder in tqdm(enumerate(glob.glob(f"{targetloc}/*"))):
        #print(folder)
        impath = f"{folder}/{folder.split('/')[-1]}.jpg"
        imname = folder.split('/')[-1]
        if tablerelative:
            fullimagegroundbox = reversetablerelativebboxes_outer(folder)
        else:
            fullimagegroundbox = torch.load(glob.glob(f"{folder}/{folder.split('/')[-1]}.pt")[0])
        img = (read_image(impath) / 255).to(device)
        output = model([img])
        output = {k: v.detach().cpu() for k, v in output[0].items()}
        # print(output['boxes'], output['boxes'][output['scores']>0.8])
        if filtering:
            output['boxes'] = output['boxes'][output['scores'] > 0.8]
        fullimagepredbox = output['boxes']

        # .......................................
        # fullimagemetrics with IoDT
        # .......................................
        prediod, tariod, IoDT_tp, IoDT_fp, IoDT_fn = calcstats_IoDT(predbox=fullimagepredbox,
                                                                    targetbox=fullimagegroundbox,
                                                                    imname=imname,
                                                                    iou_thresholds=iou_thresholds)
        # print(IoDT_tp, IoDT_fp, IoDT_fn)
        IoDT_prec, IoDT_rec, IoDT_f1, IoDT_wf1 = calcmetric(IoDT_tp, IoDT_fp, IoDT_fn, iou_thresholds=iou_thresholds)
        # print(IoDT_prec, IoDT_rec, IoDT_f1, IoDT_wf1)
        tpsum_IoDT += IoDT_tp
        fpsum_IoDT += IoDT_fp
        fnsum_IoDT += IoDT_fn
        IoDTmetrics = {"img": imname, "mean pred iod": torch.mean(prediod).item(),
                       "mean tar iod": torch.mean(tariod).item(), "wf1": IoDT_wf1.item(),
                       "prednum": fullimagepredbox.shape[0]}
        IoDTmetrics.update({f"prec@{iou_thresholds[i]}": IoDT_prec[i].item() for i in range(len(iou_thresholds))})
        IoDTmetrics.update({f"recall@{iou_thresholds[i]}": IoDT_rec[i].item() for i in range(len(iou_thresholds))})
        IoDTmetrics.update({f"f1@{iou_thresholds[i]}": IoDT_f1[i].item() for i in range(len(iou_thresholds))})
        IoDTmetrics.update({f"tp@{iou_thresholds[i]}": IoDT_tp[i].item() for i in range(len(iou_thresholds))})
        IoDTmetrics.update({f"fp@{iou_thresholds[i]}": IoDT_fp[i].item() for i in range(len(iou_thresholds))})
        IoDTmetrics.update({f"fn@{iou_thresholds[i]}": IoDT_fn[i].item() for i in range(len(iou_thresholds))})

        iodtdf = pandas.concat([iodtdf, pandas.DataFrame(IoDTmetrics, index=[n])])

        # .................................
        # fullimagemetrics with iou
        # .................................
        fullprediou, fulltargetiou, fulltp, fullfp, fullfn = calcstats(fullimagepredbox, fullimagegroundbox,
                                                                       iou_thresholds=iou_thresholds,
                                                                       imname=imname)
        # print(calcstats(fullimagepredbox, fullimagegroundbox,
        #                                                               iou_thresholds=iou_thresholds, imname=preds.split('/')[-1]), fullfp, targets[0])
        fullprec, fullrec, fullf1, fullwf1 = calcmetric(fulltp, fullfp, fullfn, iou_thresholds=iou_thresholds)
        fulltpsum += fulltp
        fullfpsum += fullfp
        fullfnsum += fullfn
        fullioulist.append(fullprediou)
        fullf1list.append(fullf1)
        fullwf1list.append(fullwf1)

        fullimagemetrics = {"img": imname, "mean pred iou": torch.mean(fullprediou).item(),
                            "mean tar iou": torch.mean(fulltargetiou).item(), "wf1": fullwf1.item(),
                            "prednum": fullimagepredbox.shape[0]}
        fullimagemetrics.update({f"prec@{iou_thresholds[i]}": fullprec[i].item() for i in range(len(iou_thresholds))})
        fullimagemetrics.update({f"recall@{iou_thresholds[i]}": fullrec[i].item() for i in range(len(iou_thresholds))})
        fullimagemetrics.update({f"f1@{iou_thresholds[i]}": fullf1[i].item() for i in range(len(iou_thresholds))})
        fullimagemetrics.update({f"tp@{iou_thresholds[i]}": fulltp[i].item() for i in range(len(iou_thresholds))})
        fullimagemetrics.update({f"fp@{iou_thresholds[i]}": fullfp[i].item() for i in range(len(iou_thresholds))})
        fullimagemetrics.update({f"fn@{iou_thresholds[i]}": fullfn[i].item() for i in range(len(iou_thresholds))})

        # ........................................
        # fullimagemetrics with alternate metric
        # ........................................
        fulltp_overlap, fullfp_overlap, fullfn_overlap = calcstats_overlap(fullimagepredbox, fullimagegroundbox,
                                                                           imname=imname)

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
            tpsum_IoDT_predonly += IoDT_tp
            fpsum_IoDT_predonly += IoDT_fp
            fnsum_IoDT_predonly += IoDT_fn
            predcount += 1

            # print("here",fullimagepredbox)
        # print(fullfp, fullimagemetrics)
        fullimagedf = pandas.concat([fullimagedf, pandas.DataFrame(fullimagemetrics, index=[n])])
        # print(fullimagedf.loc[n])
    totalfullmetrics = get_dataframe(fullfnsum, fullfpsum, fulltpsum, nopredcount=fullimagedf.shape[0] - predcount,
                                     imnum=fullimagedf.shape[0], iou_thresholds=iou_thresholds)
    partialfullmetrics = get_dataframe(fullfnsum_predonly, fullfpsum_predonly, fulltpsum_predonly,
                                       iou_thresholds=iou_thresholds)
    # print(totalfullmetrics)
    overlapprec, overlaprec, overlapf1 = calcmetric_overlap(tp=tpsum_overlap, fp=fpsum_overlap, fn=fnsum_overlap)
    totaloverlapdf = pandas.DataFrame({"f1": overlapf1, "prec": overlaprec, "recall": overlaprec}, index=["overlap"])
    overlapprec_predonly, overlaprec_predonly, overlapf1_predonly = calcmetric_overlap(tp=tpsum_overlap_predonly,
                                                                                       fp=fpsum_overlap_predonly,
                                                                                       fn=fnsum_overlap_predonly)
    predonlyoverlapdf = pandas.DataFrame({f"Number of evaluated files": overlapdf.shape[0],
                                          f"Evaluated files without predictions:": overlapdf.shape[0] - predcount,
                                          "f1": overlapf1_predonly, "prec": overlaprec_predonly,
                                          "recall": overlaprec_predonly}, index=["overlap with valid preds"])
    totaliodt = get_dataframe(fnsum=fnsum_IoDT, fpsum=fpsum_IoDT, tpsum=tpsum_IoDT,
                              nopredcount=iodtdf.shape[0] - predcount, imnum=iodtdf.shape[0],
                              iou_thresholds=iou_thresholds)
    predonlyiodt = get_dataframe(fnsum=fnsum_IoDT_predonly, fpsum=fpsum_IoDT_predonly, tpsum=tpsum_IoDT_predonly,
                                 iou_thresholds=iou_thresholds)

    conclusiondf = pandas.DataFrame(columns=["wf1"])

    conclusiondf = pandas.concat([conclusiondf, pandas.DataFrame(totalfullmetrics, index=["full image IoU"]),
                                  pandas.DataFrame(partialfullmetrics, index=["full image IoU with valid preds"]),
                                  predonlyoverlapdf, pandas.DataFrame(totaliodt, index=["full image IoDt"]),
                                  pandas.DataFrame(predonlyiodt, index=[" full image IoDt with valid preds"])])

    #print(conclusiondf)
    # save results
    #os.makedirs(saveloc, exist_ok=True)
    #print(fullimagedf.loc[50])
    #print(saveloc)
    overlapdf.to_csv(f"{saveloc}/fullimageoverlapeval.csv")
    fullimagedf.to_csv(f"{saveloc}/fullimageiou.csv")
    iodtdf.to_csv(f"{saveloc}/fullimageiodt.csv")
    conclusiondf.to_csv(f"{saveloc}/overview.csv")


def inference_tablecutout(datapath: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/test",
                          modelpath: str = f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn"
                                           f"/run3_BonnData_cell_aug_loadrun_GloSAT_cell_aug_e250_es_e250_es.pt",
                          datasetname: str = "BonnData",
                          iou_thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9], filtering=False, saveboxes=False):
    #print("here")
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, **{"box_detections_per_img": 200}
    )
    model.load_state_dict(
        torch.load(
            modelpath
        )
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        #model.eval()
    else:
        print("Cuda not available")
        return
    model.eval()
    saveloc = f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/testeval/{datasetname}"
    boxsaveloc = f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/{datasetname}"
    if filtering:
        saveloc = f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/testeval/{datasetname}/filtering"
    os.makedirs(saveloc, exist_ok=True)
    if saveboxes:
        os.makedirs(boxsaveloc, exist_ok=True)

    ioulist = []
    f1list = []
    wf1list = []
    tpsum = torch.zeros(len(iou_thresholds))
    fpsum = torch.zeros(len(iou_thresholds))
    fnsum = torch.zeros(len(iou_thresholds))
    ioudf = pandas.DataFrame(columns=["img", "mean pred iou", "mean tar iou", "wf1", "prednum"])
    #.............................
    tpsum_iodt = torch.zeros(len(iou_thresholds))
    fpsum_iodt = torch.zeros(len(iou_thresholds))
    fnsum_iodt = torch.zeros(len(iou_thresholds))
    iodtlist = []
    f1list_iodt = []
    wf1list_iodt = []
    iodtdf = pandas.DataFrame(columns=["img", "mean pred iodt", "mean tar iodt", "wf1", "prednum"])
    #print("here")
    for n, folder in enumerate(glob.glob(f"{datapath}/*")):
        #print(folder)
        for tableimg in glob.glob(f"{folder}/*table_*pt"):
            #print(glob.glob(f"{folder}/*cell_*jpg"))
            num = tableimg.split(".")[-2].split("_")[-1]
            img = (torch.load(tableimg) / 255).to(device)
            #img.to(device)
            target = torch.load(f"{folder}/{folder.split('/')[-1]}_cell_{num}.pt")
            #print(target)
            output = model([img])
            output = {k: v.detach().cpu() for k, v in output[0].items()}
            #print(output['boxes'], output['boxes'][output['scores']>0.8])
            if filtering:
                output['boxes'] = output['boxes'][output['scores'] > 0.8]

            if saveboxes:
                os.makedirs(f"{boxsaveloc}/{folder.split('/')[-1]}", exist_ok=True)
                torch.save(output['boxes'], f"{boxsaveloc}/{folder.split('/')[-1]}/{folder.split('/')[-1]}_{num}.pt")
                #print(f"{boxsaveloc}/{folder.split('/')[-1]}")

            #..................IoU
            prediou, targetiou, tp, fp, fn = calcstats(predbox=output['boxes'], targetbox=target,
                                                       iou_thresholds=iou_thresholds,
                                                       imname=folder.split('/')[-1] + "_" + num)
            prec, rec, f1, wf1 = calcmetric(tp, fp, fn, iou_thresholds=iou_thresholds)
            tpsum += tp
            fpsum += fp
            fnsum += fn
            #print(tp, fp)
            ioulist.append(prediou)
            f1list.append(f1)
            wf1list.append(wf1)
            ioumetrics = {"img": folder.split('/')[-1], "mean pred iou": torch.mean(prediou).item(),
                          "mean tar iou": torch.mean(targetiou).item(), "wf1": wf1.item(),
                          "prednum": output['boxes'].shape[0]}
            ioumetrics.update(
                {f"prec@{iou_thresholds[i]}": prec[i].item() for i in range(len(iou_thresholds))})
            ioumetrics.update(
                {f"recall@{iou_thresholds[i]}": rec[i].item() for i in range(len(iou_thresholds))})
            ioumetrics.update({f"f1@{iou_thresholds[i]}": f1[i].item() for i in range(len(iou_thresholds))})
            ioumetrics.update({f"tp@{iou_thresholds[i]}": tp[i].item() for i in range(len(iou_thresholds))})
            ioumetrics.update({f"fp@{iou_thresholds[i]}": fp[i].item() for i in range(len(iou_thresholds))})
            ioumetrics.update({f"fn@{iou_thresholds[i]}": fn[i].item() for i in range(len(iou_thresholds))})
            ioudf = pandas.concat([ioudf, pandas.DataFrame(ioumetrics, index=[f"{n}.{num}"])])
            print(ioudf)
            #...........................
            #....................IoDt
            prediodt, targetiodt, tp_iodt, fp_iodt, fn_iodt = calcstats_IoDT(predbox=output['boxes'], targetbox=target,
                                                                             iou_thresholds=iou_thresholds,
                                                                             imname=folder.split('/')[-1] + "_" + num)
            prec_iodt, rec_iodt, f1_iodt, wf1_iodt = calcmetric(tp_iodt, fp_iodt, fn_iodt,
                                                                iou_thresholds=iou_thresholds)
            # print(wf1)
            tpsum_iodt += tp_iodt
            fpsum_iodt += fp_iodt
            fnsum_iodt += fn_iodt
            iodtlist.append(prediodt)
            f1list_iodt.append(f1_iodt)
            wf1list_iodt.append(wf1_iodt)
            iodtmetrics = {"img": folder.split('/')[-1], "mean pred iodt": torch.mean(prediodt).item(),
                           "mean tar iodt": torch.mean(targetiodt).item(), "wf1": wf1_iodt.item(),
                           "prednum": output['boxes'].shape[0]}
            iodtmetrics.update(
                {f"prec@{iou_thresholds[i]}": prec_iodt[i].item() for i in range(len(iou_thresholds))})
            iodtmetrics.update(
                {f"recall@{iou_thresholds[i]}": rec_iodt[i].item() for i in range(len(iou_thresholds))})
            iodtmetrics.update({f"f1@{iou_thresholds[i]}": f1_iodt[i].item() for i in range(len(iou_thresholds))})
            iodtmetrics.update({f"tp@{iou_thresholds[i]}": tp_iodt[i].item() for i in range(len(iou_thresholds))})
            iodtmetrics.update({f"fp@{iou_thresholds[i]}": fp_iodt[i].item() for i in range(len(iou_thresholds))})
            iodtmetrics.update({f"fn@{iou_thresholds[i]}": fn_iodt[i].item() for i in range(len(iou_thresholds))})
            iodtdf = pandas.concat([iodtdf, pandas.DataFrame(iodtmetrics, index=[f"{n}.{num}"])])

    conclusiondf = pandas.DataFrame(columns=["wf1"])
    fulliodtdf = get_dataframe(fnsum=fnsum_iodt, fpsum=fpsum_iodt, tpsum=tpsum_iodt, iou_thresholds=iou_thresholds)
    fullioudf = get_dataframe(fnsum=fnsum, fpsum=fpsum, tpsum=tpsum, iou_thresholds=iou_thresholds)
    conclusiondf = pandas.concat(
        [conclusiondf, pandas.DataFrame(fullioudf, index=["IoU"]), pandas.DataFrame(fulliodtdf, index=["IoDt"])])
    ioudf.to_csv(f"{saveloc}/iou.csv")
    iodtdf.to_csv(f"{saveloc}/iodt.csv")
    conclusiondf.to_csv(f"{saveloc}/overview.csv")


if __name__ == '__main__':
    inference_fullimg()
    inference_fullimg(modelpath=f"{Path(__file__).parent.absolute()}/../../data/checkpoints/fasterrcnn/test5_with_valid_split_Tablesinthewild_fullimage_e50_es.pt")
    #pass
    #inference_fullimg(iou_thresholds=[0.9])
    for cat in glob.glob(f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/*"):
        print(cat)
        inference_fullimg(targetloc=cat, datasetname=f"Tablesinthewild/cat.split('/')[-1]")
    for cat in glob.glob(f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/*"):
        print(cat)
        inference_fullimg(targetloc=cat, datasetname=f"Tablesinthewild/cat.split('/')[-1]", modelpath=f"{Path(__file__).parent.absolute()}/../../data/checkpoints/fasterrcnn/test5_with_valid_split_Tablesinthewild_fullimage_e50_es.pt")
    inference_fullimg(targetloc=f"{Path(__file__).parent.absolute()}/../../data/BonnData/test", datasetname="BonnData", modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/BonnDataFullImage1_BonnData_fullimage_e250_es.pt", tablerelative=True)
    inference_fullimg(targetloc=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
                      datasetname="GloSat",
                      modelpath=f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/GloSatFullImage1_GloSat_fullimage_e250_es.pt",
                      tablerelative=True)
    #inference_fullimg(targetloc=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/simple", datasetname='simple')
    #inference_tablecutout(datapath=)
    #inference_tablecutout(filtering=True, saveboxes=True)
    #inference_tablecutout(datapath = f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
    #          modelpath = f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn"
    #                           f"/run_GloSAT_cell_aug_e250_es.pt",
    #          datasetname = "GloSat")
