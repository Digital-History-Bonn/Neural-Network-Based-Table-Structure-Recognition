import glob
import json
import math
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.cluster import DBSCAN
from torchvision.io import read_image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.ops import box_iou
from torchvision.utils import draw_bounding_boxes

from src.historicdocumentprocessing.kosmos_eval import calcmetric, extractboxes


def avrgeuc(boxes: torch.Tensor) -> float:
    count = 0
    dist = 0
    for box1 in boxes:
        singledist = 0
        for box2 in boxes:
            if not torch.equal(box1, box2):
                # print("j")
                new = eucsimilarity(box1.numpy(), box2.numpy())
                if not singledist or 0 < new < singledist:
                    singledist = new
            # singledist = abs(math.sqrt(pow((abs(box1[0]-box2[2])),2)+pow(abs(box1[3]-box2[1]),2)))
        #        print("f")
        dist += singledist
        count += 1
    # if dist==0:
    #    print(boxes, dist)
    # print(dist, count)
    if count != 0 and dist == 0:
        # print(boxes, dist)
        return 1
    elif count == 0:
        return 0
    return dist / count


def eucsimilarity(x, y):
    # print(x[0], y, (x[2]-x[0]))
    # print(pow(torch.norm(torch.max(torch.zeros(2), x[:2]-y[2:])),2))
    # print(pow(torch.max(torch.zeros(2), y[:2]-x[2:]),2))
    # print(np.where((x[:2]-y[2:])<0,0, (x[:2]-y[2:])), x[:2]-y[2:])
    slice = int(x.shape[0] / 2)
    # print(slice)
    res = math.sqrt(
        pow(
            np.linalg.norm(
                np.where((x[:slice] - y[slice:]) < 0, 0, (x[:slice] - y[slice:]))
            ),
            2,
        )
        + pow(
            np.linalg.norm(
                np.where((y[:slice] - x[slice:]) < 0, 0, y[:slice] - x[slice:])
            ),
            2,
        )
    )
    # res = math.sqrt(pow(np.linalg.norm(np.where((x[:2]-y[2:])<0,0, (x[:2]-y[2:]))),2)+pow(np.linalg.norm(np.where((y[:2]-x[2:])<0,0, y[:2]-x[2:])),2))
    # res = abs(math.sqrt(pow(x[2]-x[0],2)+pow((x[3]-x[1]),2))-math.sqrt(pow(y[2]-y[0],2)+pow(y[3]-y[1],2)))
    # print(x, y)
    # print(res)
    return res


def clustertables(boxes: torch.Tensor, epsprefactor: float = 1 / 6):
    tables = []
    eps = avrgeuc(boxes)
    # print("eps: ", eps)
    if eps:
        clustering = DBSCAN(
            eps=(epsprefactor) * eps, min_samples=2, metric=eucsimilarity
        ).fit(boxes.numpy())
        for label in set(clustering.labels_):
            table = boxes[clustering.labels_ == label]
            # print(label, clustering.labels_)
            tables.append(table)
    return tables


def clustertablesseperately(
    boxes: torch.Tensor,
    epsprefactor: Tuple[float, float] = tuple([3, 1.5]),
    includeoutlier: bool = True,
    minsamples: List[int] = [4, 5]
):
    tables = []
    xtables = []
    # print(boxes.shape)
    xboxes = boxes[:, [0, 2]]
    # yboxes = boxes[:,[1,3]]
    # print(boxes[:,[0,2]].shape)
    xdist = avrgeuc(xboxes)
    # ydist = avrgeuc(yboxes)
    #    print("h")
    epsprefactor1 = epsprefactor[0]
    epsprefactor2 = epsprefactor[1]
    if xdist:
        clustering = DBSCAN(
            eps=(epsprefactor1) * xdist, min_samples=minsamples[0], metric=eucsimilarity
        ).fit(xboxes.numpy())
        for label in set(clustering.labels_):
            xtable = boxes[clustering.labels_ == label]
            # print(label, clustering.labels_)
            if includeoutlier or (int(label) != -1):
                xtables.append(xtable)
            else:
                print(label)
            # print(xtable)
        for prototable in xtables:
            yboxes = prototable[:, [1, 3]]
            ydist = avrgeuc(yboxes)
            if ydist:
                clustering = DBSCAN(
                    eps=(epsprefactor2) * ydist, min_samples=minsamples[1], metric=eucsimilarity
                ).fit(yboxes.numpy())
                for label in set(clustering.labels_):
                    table = prototable[(clustering.labels_ == label)]
                    # print(label, clustering.labels_)
                    if includeoutlier or (int(label) != -1):
                        tables.append(table)
                    else:
                        print(label)
            #                    print("y")
            elif len(prototable) == 1:
                tables.append(prototable)
    return tables


def getsurroundingtable(boxes: torch.Tensor) -> torch.Tensor:
    """get surrounding table of a group of bounding boxes given as torch.Tensor
    Args:
        boxes: (cell) bounding boxes as torch.Tensor

    Returns: surrounding table"""
    return torch.hstack(
        [
            torch.min(boxes[:, 0]),
            torch.min(boxes[:, 1]),
            torch.max(boxes[:, 2]),
            torch.max(boxes[:, 3]),
        ]
    )


def gettablerelativebbox(box: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    # print(box, table)
    return torch.hstack(
        [box[0] - table[0], box[1] - table[1], box[2] - table[0], box[3] - table[1]]
    )


def gettablerelativebboxes(boxes: torch.Tensor) -> torch.Tensor:
    tablecoords = getsurroundingtable(boxes)
    # print(tablecoords)
    newboxes = []
    for box in boxes:
        newboxes.append(gettablerelativebbox(box, tablecoords))
    # print(newboxes)
    return torch.vstack(newboxes)


def BonnTablebyCat(
    categoryfile: str = f"{Path(__file__).parent.absolute()}/../../../data/BonnData/Tabellen/allinfosubset_manuell_vervollständigt.xlsx",
    resultfile: str = f"{Path(__file__).parent.absolute()}/../../../results/kosmos25/testevaltotal/BonnData_Tables/fullimageiodt.csv",
    resultmetric: str = "iodt",
):
    """
    filter bonntable eval results by category and calculate wf1 over different categories
    Args:
        categoryfile:
        resultfile:

    Returns:

    """
    df = pd.read_csv(resultfile)
    catinfo = pd.read_excel(categoryfile)
    df = df.rename(columns={"img": "Dateiname"})
    df1 = pd.merge(df, catinfo, on="Dateiname")
    subsetwf1df = {"wf1": [], "category": [], "len": [], "nopred": []}
    # replace category 1 by 2 since there are no images without tables in test dataset
    df1 = df1.replace({"category": 1}, 2)
    # print(df1.category, df1.Dateiname)
    for cat in df1.category.unique():
        if not pd.isna(cat):
            subset = df1[df1.category == cat]
            tp = []
            fp = []
            fn = []
            for i in [0.5, 0.6, 0.7, 0.8, 0.9]:
                tp.append(subset[f"tp@{str(i)}"].sum())
                fp.append(subset[f"fp@{str(i)}"].sum())
                fn.append(subset[f"fn@{str(i)}"].sum())
            # subsetwf1df['wf1'].append(subset.wf1.sum() / len(subset))
            prec, rec, f1, wf1 = calcmetric(
                tp=torch.Tensor(tp), fp=torch.Tensor(fp), fn=torch.Tensor(fn)
            )
            subsetwf1df["wf1"].append(wf1.item())
            subsetwf1df["category"].append(cat)
            subsetwf1df["len"].append(len(subset))
            # print(subset[subset['prednum'].eq(0) == True])
            subsetwf1df["nopred"].append(len(subset[subset["prednum"].eq(0) == True]))
    if len(df1[pd.isna(df1.category)]) > 0:
        subset = df1[pd.isna(df1.category)]
        subsetwf1df["category"].append("no category")
        subsetwf1df["len"].append(len(subset))
        subsetwf1df["wf1"].append(subset.wf1.sum() / len(subset))
        subsetwf1df["nopred"].append(len(subset[subset["nopred"].eq(0) == True]))
    saveloc = f"{'/'.join(resultfile.split('/')[:-1])}/{resultmetric}_bycategory.xlsx"
    pd.DataFrame(subsetwf1df).set_index("category").to_excel(saveloc)


if __name__ == "__main__":
    """
    BonnTablebyCat(resultfile=f"{Path(__file__).parent.absolute()}/../../../results/kosmos25/testevalfinal1/BonnData_Tables/iou_0.5_0.9/tableareaonly/fullimageiodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/kosmos25/testevalfinal1/BonnData_Tables/iou_0.5_0.9/tableareaonly/fullimageiou.csv",
        resultmetric="iou")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/fullimg/BonnData/BonnDataFullImage1_BonnData_fullimage_e250_es.pt/tableareaonly/iou_0.5_0.9/fullimageiodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/fullimg/BonnData/BonnDataFullImage1_BonnData_fullimage_e250_es.pt/iou_0.5_0.9/fullimageiodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/fullimg/BonnData/BonnDataFullImage1_BonnData_fullimage_e250_es.pt/tableareaonly/iou_0.5_0.9/fullimageiou.csv",
        resultmetric="iou")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/fullimg/BonnData/BonnDataFullImage1_BonnData_fullimage_e250_es.pt/iou_0.5_0.9/fullimageiou.csv",
        resultmetric="iou")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_es.pt/tableareaonly/iou_0.5_0.9/fullimageiodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_es.pt/iou_0.5_0.9/fullimageiodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_es.pt/tableareaonly/iou_0.5_0.9/fullimageiou.csv",
        resultmetric="iou")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_es.pt/iou_0.5_0.9/fullimageiou.csv",
        resultmetric="iou")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_end.pt/tableareaonly/iou_0.5_0.9/fullimageiodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_end.pt/iou_0.5_0.9/fullimageiodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_end.pt/tableareaonly/iou_0.5_0.9/fullimageiou.csv",
        resultmetric="iou")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_end.pt/iou_0.5_0.9/fullimageiou.csv",
        resultmetric="iou")


    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/tableareacutout/BonnData/run3_BonnData_cell_aug_loadrun_GloSAT_cell_aug_e250_es_e250_es.pt/iodt.csv")

    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/tableareacutout/BonnData/run3_BonnData_cell_loadrun_GloSAT_cell_e250_es_e250_es.pt/iodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/tableareacutout/BonnData/run_BonnData_cell_e250_es.pt/iodt.csv")

    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/tableareacutout/BonnData/run3_BonnData_cell_aug_loadrun_GloSAT_cell_aug_e250_es_e250_es.pt/iou.csv",
        resultmetric="iou")

    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/tableareacutout/BonnData/run3_BonnData_cell_loadrun_GloSAT_cell_e250_es_e250_es.pt/iou.csv",
        resultmetric="iou")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/tableareacutout/BonnData/run_BonnData_cell_e250_es.pt/iou.csv",
        resultmetric="iou")


    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, **{"box_detections_per_img": 200}
    )
    model.load_state_dict(
        torch.load(
            f"{Path(__file__).parent.absolute()}/../../../checkpoints/fasterrcnn/BonnDataFullImage1_BonnData_fullimage_e250_es.pt"
        )
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        model.eval()
    else:
        print("Cuda not available")
        exit()
    #boxes = torch.load(f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/BonnData/I_HA_Rep_89_Nr_16160_0170/I_HA_Rep_89_Nr_16160_0170.pt")
    img = (read_image(f"{Path(__file__).parent.absolute()}/../../../data/BonnData/test/I_HA_Rep_89_Nr_16160_0170/I_HA_Rep_89_Nr_16160_0170.jpg") / 255).to(device)
    output = model([img])
    output = {k: v.detach().cpu() for k, v in output[0].items()}
    boxes= output['boxes']
    tables = clustertables(boxes, epsprefactor=1/6)
    img = read_image(
        f"{Path(__file__).parent.absolute()}/../../../data/BonnData/test/I_HA_Rep_89_Nr_16160_0170/I_HA_Rep_89_Nr_16160_0170.jpg")
    for i,t in enumerate(tables):
       res = draw_bounding_boxes(image=img, boxes=t)
       res = Image.fromarray(res.permute(1, 2, 0).numpy())
       # print(f"{savepath}/{identifier}.jpg")
       res.save(f"{Path(__file__).parent.absolute()}/../../../images/test/test_rcnn_{i}.jpg")

    """
    with open(
        f"{Path(__file__).parent.absolute()}/../../../results/kosmos25/BonnData/Tabellen/test/I_HA_Rep_89_Nr_16160_0170/I_HA_Rep_89_Nr_16160_0170.jpg.json"
    ) as p:
        boxes = extractboxes(json.load(p))
    # tables = clustertables(boxes)
    tables = clustertablesseperately(boxes, includeoutlier=False)
    img = read_image(
        f"{Path(__file__).parent.absolute()}/../../../data/BonnData/test/I_HA_Rep_89_Nr_16160_0170/I_HA_Rep_89_Nr_16160_0170.jpg"
    )
    for i, t in enumerate(tables):
        res = draw_bounding_boxes(image=img, boxes=t)
        res = Image.fromarray(res.permute(1, 2, 0).numpy())
        # print(f"{savepath}/{identifier}.jpg")
        res.save(
            f"{Path(__file__).parent.absolute()}/../../../images/test/test_{i}.jpg"
        )

    """"
    BonnTablebyCat()
    BonnTablebyCat(resultfile=f"{Path(__file__).parent.absolute()}/../../../results/kosmos25/testevaltotal/BonnData_Tables/fullimageiou.csv", resultmetric="iou")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/tableareacutout/BonnData/projectgroupmodelinference/iodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/tableareacutout/BonnData/run3_BonnData_cell_aug_loadrun_GloSAT_cell_aug_e250_es_e250_es.pt/iodt.csv")

    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/tableareacutout/BonnData/run3_BonnData_cell_loadrun_GloSAT_cell_e250_es_e250_es.pt/iodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/tableareacutout/BonnData/run_BonnData_cell_e250_es.pt/iodt.csv")

    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/tableareacutout/BonnData/run3_BonnData_cell_aug_loadrun_GloSAT_cell_aug_e250_es_e250_es.pt/iou.csv", resultmetric="iou")

    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/tableareacutout/BonnData/run3_BonnData_cell_loadrun_GloSAT_cell_e250_es_e250_es.pt/iou.csv", resultmetric="iou")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/tableareacutout/BonnData/run_BonnData_cell_e250_es.pt/iou.csv", resultmetric="iou")

    BonnTablebyCat(resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/tableareacutout/BonnData/projectgroupmodelinference/iou.csv", resultmetric="iou")
    BonnTablebyCat(resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/fullimg/BonnData/BonnDataFullImage1_BonnData_fullimage_e250_es.pt/tableareaonly/iou_0.5_0.9/fullimageiodt.csv")
    BonnTablebyCat(resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/fullimg/BonnData/BonnDataFullImage1_BonnData_fullimage_e250_es.pt/iou_0.5_0.9/fullimageiodt.csv")
    BonnTablebyCat(resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/fullimg/BonnData/BonnDataFullImage1_BonnData_fullimage_e250_es.pt/tableareaonly/iou_0.5_0.9/fullimageiou.csv", resultmetric="iou")
    BonnTablebyCat(resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/fullimg/BonnData/BonnDataFullImage1_BonnData_fullimage_e250_es.pt/iou_0.5_0.9/fullimageiou.csv", resultmetric="iou")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_es.pt/tableareaonly/iou_0.5_0.9/fullimageiodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_es.pt/iou_0.5_0.9/fullimageiodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_es.pt/tableareaonly/iou_0.5_0.9/fullimageiou.csv",
        resultmetric="iou")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_es.pt/iou_0.5_0.9/fullimageiou.csv",
        resultmetric="iou")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_end.pt/tableareaonly/iou_0.5_0.9/fullimageiodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_end.pt/iou_0.5_0.9/fullimageiodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_end.pt/tableareaonly/iou_0.5_0.9/fullimageiou.csv",
        resultmetric="iou")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_end.pt/iou_0.5_0.9/fullimageiou.csv",
        resultmetric="iou")
    """


def remove_invalid_bbox(box, impath: str = "") -> torch.Tensor:
    newbox = []
    for b in box:
        if b[0] < b[2] and b[1] < b[3]:
            # newbox= torch.vstack((newbox, b.clone()))
            newbox.append(b)
        else:
            print(f"Invalid bounding box in image {impath.split('/')[-1]}", b)
    # print(newbox)
    return torch.vstack(newbox) if newbox else torch.empty(0, 4)
