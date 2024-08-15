import glob
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

from src.historicdocumentprocessing.kosmos_eval import extractboxes, reversetablerelativebboxes_outer
from src.historicdocumentprocessing.util.tablesutil import clustertables, clustertablesseperately


def postprocess_kosmos(targetloc: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData/test", predloc:str = f"{Path(__file__).parent.absolute()}/../../results/kosmos25/BonnData"
                                      f"/Tabellen/test", tableavail:bool = True, tableonly:bool = True):
    for n, targets in tqdm(enumerate(glob.glob(f"{targetloc}/*"))):
        preds = glob.glob(f"{predloc}/{targets.split('/')[-1]}")[0]
        # imagepred = [file for file in glob.glob(f"{preds}/*.json") if "_table_" not in file][0]
        # bboxes on full image
        # fullimagepred = glob.glob(f"{preds}/(?!.*_table_)^.*$")[0]
        # print(preds, targets)
        fullimagepred = [file for file in glob.glob(f"{preds}/*") if "_table_" not in file][0]
        # print(fullimagepred)
        with open(fullimagepred) as p:
            fullimagepredbox = extractboxes(json.load(p), fpath=targets if (tableonly and tableavail) else None)
            # print(targets[0])
            for epsprefactor in [1]:
                clusteredtables = clustertablesseperately(fullimagepredbox, epsprefactor=epsprefactor)
                imname = preds.split('/')[-1]
                if tableavail and clusteredtables:
                    tablenum = len(glob.glob(f"{targets}/*cell*"))
                    fullimagegroundbox = reversetablerelativebboxes_outer(targets)
                    if tablenum!=len(clusteredtables):
                        print("imname: ",imname, epsprefactor, tablenum, len(clusteredtables))
                        for i, t in enumerate(clusteredtables):
                            img = read_image(glob.glob(f"{targets}/*jpg")[0])
                            res = draw_bounding_boxes(image=img, boxes=t)
                            res = Image.fromarray(res.permute(1, 2, 0).numpy())
                            # print(f"{savepath}/{identifier}.jpg")
                            res.save(f"{Path(__file__).parent.absolute()}/../../images/test/test_kosmos_{imname}_{i}.jpg")
                else:
                    fullimagegroundbox = torch.load(
                        glob.glob(f"{targetloc}/{preds.split('/')[-1]}/{preds.split('/')[-1]}.pt")[0])

if __name__=='__main__':
    #postprocess_kosmos()
    postprocess_kosmos(targetloc=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
                       predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/GloSat/test1")

