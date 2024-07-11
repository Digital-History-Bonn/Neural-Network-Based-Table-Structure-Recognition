import glob
import os
from pathlib import Path
from typing import List

import pandas
import torch

from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image

from kosmos_eval import calcstats, calcmetric
from src.historicdocumentprocessing.kosmos_eval import calcstats_IoDT, get_dataframe


def inference(datapath: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/test",
              modelpath: str = f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn"
                               f"/run3_BonnData_cell_aug_loadrun_GloSAT_cell_aug_e250_es_e250_es.pt",
              datasetname: str = "BonnData",
              iou_thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9]):
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
    #os.makedirs(saveloc, exist_ok=True)
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
            img = (torch.load(tableimg) / 256).to(device)
            #img.to(device)
            target = torch.load(f"{folder}/{folder.split('/')[-1]}_cell_{num}.pt")
            #print(target)
            output = model([img])
            output = {k: v.detach().cpu() for k, v in output[0].items()}
            #print(output)
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
            #print(ioudf)
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
            iodtmetrics = {"img": folder.split('/')[-1], "mean pred iodt": torch.mean(prediou).item(),
                           "mean tar iodt": torch.mean(targetiou).item(), "wf1": wf1.item(),
                           "prednum": output['boxes'].shape[0]}
            iodtmetrics.update(
                {f"prec@{iou_thresholds[i]}": prec[i].item() for i in range(len(iou_thresholds))})
            iodtmetrics.update(
                {f"recall@{iou_thresholds[i]}": rec[i].item() for i in range(len(iou_thresholds))})
            iodtmetrics.update({f"f1@{iou_thresholds[i]}": f1[i].item() for i in range(len(iou_thresholds))})
            iodtmetrics.update({f"tp@{iou_thresholds[i]}": tp[i].item() for i in range(len(iou_thresholds))})
            iodtmetrics.update({f"fp@{iou_thresholds[i]}": fp[i].item() for i in range(len(iou_thresholds))})
            iodtmetrics.update({f"fn@{iou_thresholds[i]}": fn[i].item() for i in range(len(iou_thresholds))})
            iodtdf = pandas.concat([iodtdf, pandas.DataFrame(iodtmetrics, index=[f"{n}.{num}"])])
    conclusiondf = pandas.DataFrame(columns=["wf1"])
    fulliodtdf = get_dataframe(fnsum=fnsum_iodt, fpsum=fpsum_iodt, tpsum=tpsum_iodt, iou_thresholds=iou_thresholds)
    fullioudf = get_dataframe(fnsum=fnsum, fpsum=fpsum, tpsum=tpsum, iou_thresholds=iou_thresholds)
    conclusiondf = pandas.concat([conclusiondf, pandas.DataFrame(fullioudf, index=["IoU"]), pandas.DataFrame(fulliodtdf, index=["IoDt"])])
    ioudf.to_csv(f"{saveloc}/iou.csv")
    iodtdf.to_csv(f"{saveloc}/iodt.csv")
    conclusiondf.to_csv(f"{saveloc}/overview.csv")

if __name__ == '__main__':
    inference()
