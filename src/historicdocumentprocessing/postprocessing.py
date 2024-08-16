import glob
import json
import os
from pathlib import Path

import torch
from PIL import Image
from torchaudio.functional import filtering
from torchvision.io import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

from src.historicdocumentprocessing.fasterrcnn_eval import tableareabboxes
from src.historicdocumentprocessing.kosmos_eval import extractboxes, reversetablerelativebboxes_outer
from src.historicdocumentprocessing.util.tablesutil import clustertables, clustertablesseperately, remove_invalid_bbox


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
            fullimagepredbox = remove_invalid_bbox(extractboxes(json.load(p), fpath=targets if (tableonly and tableavail) else None))
            # print(targets[0])
            #print("here")
            postprocess(fullimagepredbox, preds, tableavail, targetloc, targets)


def postprocess(fullimagepredbox, preds, tableavail, targetloc, targets):
    for epsprefactor in [1]:
        clusteredtables = clustertablesseperately(fullimagepredbox, epsprefactor=epsprefactor)
        #print(len(clusteredtables))
        imname = preds.split('/')[-1]
        if tableavail and clusteredtables:
            tablenum = len(glob.glob(f"{targets}/*cell*"))
            fullimagegroundbox = reversetablerelativebboxes_outer(targets)
            if tablenum != len(clusteredtables):
                print("imname: ", imname, epsprefactor, tablenum, len(clusteredtables))
                for i, t in enumerate(clusteredtables):
                    if t.shape[0]>1:
                        img = read_image(glob.glob(f"{targets}/*jpg")[0])
                        colors = ["green" for i in range(t.shape[0])]
                        res = draw_bounding_boxes(image=img, boxes=t, colors=colors)
                        res = Image.fromarray(res.permute(1, 2, 0).numpy())
                        # print(f"{savepath}/{identifier}.jpg")
                        res.save(f"{Path(__file__).parent.absolute()}/../../images/test/test_kosmos1_{imname}_{i}.jpg")
        else:
            fullimagegroundbox = torch.load(
                glob.glob(f"{targetloc}/{preds.split('/')[-1]}/{preds.split('/')[-1]}.pt")[0])


def postprocess_rcnn(tableareaonly=None, modelpath=None, targetloc=None, tablerelative=None):
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
        if tableareaonly:
            output['boxes'] = tableareabboxes(output['boxes'], folder)
        fullimagepredbox = output['boxes']

if __name__=='__main__':
    #postprocess_kosmos()
    postprocess_kosmos(targetloc=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
                       predloc=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/GloSat/test1")

