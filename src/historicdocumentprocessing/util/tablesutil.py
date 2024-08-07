import json
import math
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
import torch
from tensorboard.summary.v1 import image
from torchvision.io import read_image

from sklearn.cluster import DBSCAN

import numpy as np

from src.historicdocumentprocessing.kosmos_eval import extractboxes
from torchvision.utils import draw_bounding_boxes
from PIL import Image


def avrgeuc(boxes: torch.Tensor)->float:
    count = 0
    dist =  0
    for box1 in boxes:
        singledist = 0
        for box2 in boxes:
            if not torch.equal(box1,box2):
                singledist = eucsimilarity(box1.numpy(),box2.numpy())
                dist+=singledist
                count+=1
            #singledist = abs(math.sqrt(pow((abs(box1[0]-box2[2])),2)+pow(abs(box1[3]-box2[1]),2)))
    print(dist, count)
    return dist/count

def eucsimilarity(x, y):
    #print(x[0], y, (x[2]-x[0]))
    #print(pow(torch.norm(torch.max(torch.zeros(2), x[:2]-y[2:])),2))
    #print(pow(torch.max(torch.zeros(2), y[:2]-x[2:]),2))
    #print(np.where((x[:2]-y[2:])<0,0, (x[:2]-y[2:])), x[:2]-y[2:])
    res = math.sqrt(pow(np.linalg.norm(np.where((x[:2]-y[2:])<0,0, (x[:2]-y[2:]))),2)+pow(np.linalg.norm(np.where((y[:2]-x[2:])<0,0, y[:2]-x[2:])),2))
    #res = abs(math.sqrt(pow(x[2]-x[0],2)+pow((x[3]-x[1]),2))-math.sqrt(pow(y[2]-y[0],2)+pow(y[3]-y[1],2)))
    #print(x, y)
    print(res)
    return res

def clustertables(boxes:torch.Tensor):
    tables = []
    eps = avrgeuc(boxes)
    print("eps: ", eps)
    clustering = DBSCAN(eps=(1/6)*eps, min_samples=2, metric=eucsimilarity).fit(boxes.numpy())
    for label in set(clustering.labels_):
        table = boxes[clustering.labels_==label]
        print(label, clustering.labels_)
        tables.append(table)
    return tables


def BonnTablebyCat(categoryfile: str = f"{Path(__file__).parent.absolute()}/../../../data/BonnData/Tabellen/allinfosubset_manuell_vervollständigt.xlsx",
                   resultfile: str = f"{Path(__file__).parent.absolute()}/../../../results/kosmos25/testevaltotal/BonnData_Tables/fullimageiodt.csv", resultmetric: str = "iodt"):
    """
    filter bonntable eval results by category and calculate wf1 over different categories
    Args:
        categoryfile:
        resultfile:

    Returns:

    """
    df = pd.read_csv(resultfile)
    catinfo = pd.read_excel(categoryfile)
    df = df.rename(columns={'img': 'Dateiname'})
    df1 = pd.merge(df, catinfo, on='Dateiname')
    subsetwf1df = {'wf1': [], 'category': [], 'len': []}
    #replace category 1 by 2 since there are no images without tables in test dataset
    df1 = df1.replace({'category':1},2)
    #print(df1.category, df1.Dateiname)
    for cat in df1.category.unique():
        if not pd.isna(cat):
            subset = df1[df1.category == cat]
            subsetwf1df['wf1'].append(subset.wf1.sum() / len(subset))
            subsetwf1df['category'].append(cat)
            subsetwf1df['len'].append(len(subset))
    if len(df1[pd.isna(df1.category)]) > 0:
        subset = df1[pd.isna(df1.category)]
        subsetwf1df['category'].append('no category')
        subsetwf1df['len'].append(len(subset))
        subsetwf1df['wf1'].append(subset.wf1.sum() / len(subset))
    #print(subsetwf1df)
    saveloc = f"{'/'.join(resultfile.split('/')[:-1])}/{resultmetric}_bycategory.xlsx"
    pd.DataFrame(subsetwf1df).set_index('category').to_excel(saveloc)

if __name__ == '__main__':
    with open(f"{Path(__file__).parent.absolute()}/../../../results/kosmos25/BonnData/Tabellen/test/I_HA_Rep_89_Nr_16160_0170/I_HA_Rep_89_Nr_16160_0170.jpg.json") as p:
        boxes = extractboxes(json.load(p))
    tables = clustertables(boxes)
    img = read_image(f"{Path(__file__).parent.absolute()}/../../../data/BonnData/test/I_HA_Rep_89_Nr_16160_0170/I_HA_Rep_89_Nr_16160_0170.jpg")
    for i,t in enumerate(tables):
        res = draw_bounding_boxes(image=img, boxes=t)
        res = Image.fromarray(res.permute(1, 2, 0).numpy())
        # print(f"{savepath}/{identifier}.jpg")
        res.save(f"{Path(__file__).parent.absolute()}/../../../images/test/test_{i}.jpg")
    """
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

