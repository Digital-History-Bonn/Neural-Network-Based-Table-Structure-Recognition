import glob
import json
import os
from pathlib import Path
from typing import List

import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.io import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

from src.historicdocumentprocessing.kosmos_eval import (
    extractboxes,
    reversetablerelativebboxes_outer,
)
from src.historicdocumentprocessing.util.tablesutil import remove_invalid_bbox
from src.historicdocumentprocessing.postprocessing import postprocess

def drawimg_allmodels(datasetname:str, imgpath:str, rcnnmodels:List[str], datasetaddon_pred:str=None, filter:int=0.7):
    imname = imgpath.split("/")[-1].split(".")[-2]
    groundpath = f"{Path(__file__).parent.absolute()}/../../data/{datasetname.split('/')[-2]}/preprocessed/{datasetname.split('/')[-1]}" if "/" in datasetname else f"{Path(__file__).parent.absolute()}/../../data/{datasetname}/test"
    groundpath = f"{groundpath}/{imname}"
    predpath_kosmos = f'{f"{Path(__file__).parent.absolute()}/../../results/kosmos25/{datasetname}/{imname}/{imname}" if not datasetaddon_pred else f"{Path(__file__).parent.absolute()}/../../results/kosmos25/{datasetname}/{datasetaddon_pred}/{imname}/{imname}"}.jpg.json'
    predpath_kosmos_postprocessed = f"{Path(__file__).parent.absolute()}/../../results/kosmos25/postprocessed/fullimg/{datasetname}/eps_3.0_1.5/withoutoutlier_excludeoutliers_glosat/{imname}/{imname}.pt"
    predpath_rcnn_postprocessed = f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/postprocessed/fullimg/{datasetname}"
    for name, path in zip(['not_postprocessed', 'postprocessed'], [predpath_kosmos, predpath_kosmos_postprocessed]):
        savepath = f"{Path(__file__).parent.absolute()}/../../images/kosmos25/{name}"
        drawimg_varformat(impath=imgpath, predpath=path, groundpath=groundpath, tableonly=False, savepath=savepath)
    for model in glob.glob(f"{predpath_rcnn_postprocessed}/*"):
        if 'end' not in model:
            savepath = f"{Path(__file__).parent.absolute()}/../../images/fasterrcnn/{model.split('/')[-1]}/postprocessed"
            try:
                drawimg_varformat(impath=imgpath, predpath=f"{model}/eps_3.0_1.5/withoutoutlier_excludeoutliers_glosat/{imname}/{imname}.pt", groundpath=groundpath, tableonly=False, savepath=savepath)
            except FileNotFoundError:
                print(f"{imname} has no valid predicitons with model {model}")
                pass
    for model in rcnnmodels:
        modelpath = f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/{model}"
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
        img = (read_image(imgpath) / 255).to(device)
        output = model([img])
        output = {k: v.detach().cpu() for k, v in output[0].items()}
        fullimagepredbox_filtered = remove_invalid_bbox(output["boxes"][output["scores"]>filter])
        fullimagepredbox = remove_invalid_bbox(output["boxes"])
        drawimg_varformat_inner(impath=imgpath, box=fullimagepredbox, groundpath=groundpath, savepath=f"{Path(__file__).parent.absolute()}/../../images/fasterrcnn/{modelpath.split('/')[-1]}")
        drawimg_varformat_inner(impath=imgpath, box=fullimagepredbox_filtered, groundpath=groundpath,
                            savepath=f"{Path(__file__).parent.absolute()}/../../images/fasterrcnn/{modelpath.split('/')[-1]}/filtered_{filter}")
        filtered_processed = postprocess(fullimagepredbox_filtered, imname=imname, saveloc= "{Path(__file__).parent.absolute()}/../../images/test.pt", minsamples=[2, 3])
        drawimg_varformat_inner(impath=imgpath, box=filtered_processed, groundpath=groundpath,
                            savepath=f"{Path(__file__).parent.absolute()}/../../images/fasterrcnn/{modelpath.split('/')[-1]}/filtered_{filter}_filtertest")

def drawimg_varformat(
    impath: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData/test/Konflikttabelle.jpg",
    predpath: str = f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Konflikttabelle.jpg.json",
    groundpath: str = None,
    savepath: str = None,
    tableonly: bool = True,
):
    # img = plt.imread(impath)
    # img.setflags(write=1)
    # img = img.copy()
    # print(img.flags)
    print(groundpath)
    if "json" in predpath:
        with open(predpath) as s:
            box = extractboxes(json.load(s), groundpath if tableonly else None)
            # print(groundpath)
            # print(box)
    elif "pt" in predpath:
        box = torch.load(predpath)
    else:
        box = reversetablerelativebboxes_outer(predpath)

    labels = ["Pred" for i in range(box.shape[0])]
    colors = ["green" for i in range(box.shape[0])]
    # with open(predpath) as s:
    #    box = extractboxes(json.load(s))
    # print(img.shape)
    # print(box, box.numel())
    if not box.numel():
        print(predpath.split("/")[-1])

    drawimg_varformat_inner(box, groundpath, impath, savepath)
    # plt.show()


def drawimg_varformat_inner(box, groundpath, impath, savepath):
    img = read_image(impath)
    labels = ["Pred" for i in range(box.shape[0])]
    colors = ["green" for i in range(box.shape[0])]
    if groundpath:
        if "json" in groundpath:
            with open(groundpath) as s:
                gbox = extractboxes(json.load(s))
        elif "pt" in groundpath:
            gbox = torch.load(groundpath)
        else:
            gbox = reversetablerelativebboxes_outer(groundpath)
        labels.extend(["Ground"] * gbox.shape[0])
        colors.extend(["red"] * gbox.shape[0])
        # print(gbox)
        box = torch.vstack((box, gbox))
    # print(labels, colors, box)
    # print(gbox.shape, box.shape)
    # print(img.shape)
    # print(predpath)
    # print(gbox)
    # newbox = remove_invalid_bbox(box, impath)
    try:
        image = draw_bounding_boxes(image=img, boxes=box, labels=labels, colors=colors)
    except ValueError:
        # print(predpath)
        # print(box)
        newbox = remove_invalid_bbox(box, impath)
        labels = ["Pred" for i in range(newbox.shape[0])]
        image = draw_bounding_boxes(
            image=img, boxes=newbox, labels=labels, colors=colors
        )
    # plt.imshow(image.permute(1, 2, 0))
    if savepath:
        os.makedirs(savepath, exist_ok=True)
        identifier = impath.split("/")[-1].split(".")[-2]
        # plt.savefig(f"{savepath}/{identifier}")
        # cv2.imwrite(img=image.numpy(), filename=f"{savepath}/{identifier}.jpg")
        img = Image.fromarray(image.permute(1, 2, 0).numpy())
        # print(f"{savepath}/{identifier}.jpg")
        img.save(f"{savepath}/{identifier}.jpg")


def drawimg(
    impath: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData/test/Konflikttabelle.jpg",
    jsonpath: str = f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Konflikttabelle.jpg.json",
    savepath: str = None,
):
    # img = plt.imread(impath)
    # img.setflags(write=1)
    # img = img.copy()
    # print(img.flags)
    img = read_image(impath)
    with open(jsonpath) as s:
        box = extractboxes(json.load(s))
    # print(img.shape)
    image = draw_bounding_boxes(image=img, boxes=box)
    plt.imshow(image.permute(1, 2, 0))
    if savepath:
        os.makedirs(savepath, exist_ok=True)
        identifier = impath.split("/")[-1].split(".")[-2]
        # plt.savefig(f"{savepath}/{identifier}")
        # cv2.imwrite(img=image.numpy(), filename=f"{savepath}/{identifier}.jpg")
        img = Image.fromarray(image.permute(1, 2, 0).numpy())
        img.save(f"{savepath}/{identifier}.jpg")
    plt.show()


def drawimages(
    impath: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData/test",
    jsonpath: str = f"{Path(__file__).parent.absolute()}/../../results/kosmos25/BonnData/test",
    savepath: str = None,
    groundpath: str = None,
):
    jsons = glob.glob(f"{jsonpath}/*.json")
    for json in tqdm(jsons):
        signifier = json.split("/")[-1].split(".")[-3]
        img = glob.glob(f"{impath}/{signifier}.jpg")
        drawimg_varformat(
            impath=img[0], predpath=json, savepath=savepath, groundpath=groundpath
        )


def drawimages_var(
    impath: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData/test",
    jsonpath: str = f"{Path(__file__).parent.absolute()}/../../results/kosmos25/BonnData/test",
    savepath: str = None,
    groundpath: str = None,
    tableonly: bool = True,
    range: int = None,
):
    jsons = glob.glob(f"{jsonpath}/*.json")
    if len(jsons) == 0:
        jsons = glob.glob(f"{jsonpath}/*")
        # print(len(jsons), jsons)
    for json in tqdm(jsons):
        # print(json)
        if "json" in json:
            signifier = json.split("/")[-1].split(".")[-3]
            # img = glob.glob(f"{impath}/{signifier}.jpg")
            imgpred = json
        else:
            signifier = json.split("/")[-1]
            # print(glob.glob(f"{json}/*"),glob.glob(f"{json}/(?!.*_table_)*$"))
            # imgpred = glob.glob(f"{json}/(?!.*_table_)^.*$")[0]
            imgpred = [
                file for file in glob.glob(f"{json}/*") if "_table_" not in file
            ][0]
        # signifier = json.split("/")[-1].split(".")[-3]
        # print(signifier, imgpred)
        img = glob.glob(f"{impath}/{signifier}/{signifier}.jpg")
        # print(glob.glob(f"{impath}/{signifier}/*"))
        # print(signifier, img)
        drawimg_varformat(
            impath=img[0],
            predpath=imgpred,
            savepath=savepath,
            groundpath=(
                groundpath
                if not groundpath
                else glob.glob(f"{groundpath}/{signifier}/{signifier}*.pt")[0]
            ),
            tableonly=tableonly,
        )
        if range:
            range -= 1
            if range <= 0:
                return
        # if groundpath:
        #    groundpathfiner = glob.glob(f"{groundpath}/{signifier}")[0]
        #    #print(glob.glob(groundpathfiner), groundpathfiner)
        #    #print(f"{groundpath}/{signifier}", groundpath)
        #    drawimg_varformat(impath=img[0], predpath=imgpred, savepath=savepath, groundpath=groundpathfiner)
        # else:
        #    drawimg_varformat(impath=img[0], predpath=imgpred, savepath=savepath, groundpath=groundpath)


def main():
    pass


if __name__ == "__main__":
    drawimg_allmodels(datasetname="BonnData", imgpath=f"{Path(__file__).parent.absolute()}/../../data/BonnData/preprocessed/I_HA_Rep_89_Nr_16160_0227/I_HA_Rep_89_Nr_16160_0227.jpg", datasetaddon_pred="Tabellen/test", rcnnmodels=['BonnDataFullImage1_BonnData_fullimage_e250_es.pt','BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_es.pt', 'BonnDataFullImage_random_BonnData_fullimage_e250_random_init_es.pt'])
    # drawimg_varformat(savepath=f"{Path(__file__).parent.absolute()}/../../images/rcnn/BonnTables/test",
    #                  impath=f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/test/IMG_20190821_132903/IMG_20190821_132903_table_0.jpg",
    #                  groundpath=f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/test/IMG_20190821_132903/IMG_20190821_132903_cell_0.pt",
    #                  predpath=f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/BonnData/IMG_20190821_132903/IMG_20190821_132903.pt")
    #drawimg_varformat(
    #    impath=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/curved/table_spider_00909/table_spider_00909.jpg",
    #    groundpath=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/curved/table_spider_00909/table_spider_00909.pt",
    #    predpath=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/curved/table_spider_00909/table_spider_00909.jpg.json",
    #    savepath=f"{Path(__file__).parent.absolute()}/../../images/test/tablespider",
    #    tableonly=False,
    #)
    # drawimages_var(impath=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/overlaid",
    #                  groundpath=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/overlaid",
    #                  jsonpath=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/overlaid",
    #                  savepath=f"{Path(__file__).parent.absolute()}/../../images/Kosmos/Tablesinthewild/overlaid", tableonly=False, range=50)
    # drawimg_varformat(impath=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/simple/0hZg6EpcTdKXk6j44umj3gAAACMAAQED/0hZg6EpcTdKXk6j44umj3gAAACMAAQED.jpg",
    #                  groundpath=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/simple/0hZg6EpcTdKXk6j44umj3gAAACMAAQED/0hZg6EpcTdKXk6j44umj3gAAACMAAQED.pt",
    #                  predpath=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/simple/0hZg6EpcTdKXk6j44umj3gAAACMAAQED/0hZg6EpcTdKXk6j44umj3gAAACMAAQED.jpg.json",
    #                  savepath=f"{Path(__file__).parent.absolute()}/../../images/Kosmos/Tablesinthewild/simple", tableonly=False)
    # drawimages(savepath=f"{Path(__file__).parent.absolute()}/../../images/Kosmos")
    # drawimages_var()
    # drawimages_var(impath=f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/test",
    #               groundpath=f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/test",
    #               jsonpath=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/BonnData/Tabellen/test",
    #               savepath=f"{Path(__file__).parent.absolute()}/../../images/Kosmos/tables/tableonly")
    # drawimages_var(impath=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
    #               groundpath=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
    #               jsonpath=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/GloSat/test1",
    #               savepath=f"{Path(__file__).parent.absolute()}/../../images/Kosmos/GloSat")
    # print(torch.load(f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/preprocessed/I_HA_Rep_89_Nr_16160_0241/I_HA_Rep_89_Nr_16160_0241_tables.pt"))
    # drawimg(impath= f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/preprocessed/I_HA_Rep_89_Nr_16160_0089/I_HA_Rep_89_Nr_16160_0089.jpg", jsonpath=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/I_HA_Rep_89_Nr_16160_0089.jpg.json", savepath=f"{Path(__file__).parent.absolute()}/../../images/Kosmos")
    # drawimg(impath= f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/preprocessed/I_HA_Rep_89_Nr_16160_0089/I_HA_Rep_89_Nr_16160_0089_table_0.jpg",jsonpath=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/I_HA_Rep_89_Nr_16160_0089_table_0.jpg.json",savepath=f"{Path(__file__).parent.absolute()}/../../images/Kosmos")
    # drawimg(jsonpath= f"{Path(__file__).parent.absolute()}/../../results/kosmos25/EngNewspaper/sn83030313_2474-3224_68.jpg.json", impath=f"{Path(__file__).parent.absolute()}/../../data/EngNewspaper/raw/sn83030313_2474-3224_68.jpg", savepath=f"{Path(__file__).parent.absolute()}/../../images/Kosmos")
    # drawimg(jsonpath=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/BonnData/Newspaper/Koelnische_Zeitung_1866-06_1866-09_0046.jpg.json", impath=f"{Path(__file__).parent.absolute()}/../../data/BonnData/Koelnische_Zeitung_1866-06_1866-09_0046.jpg", savepath=f"{Path(__file__).parent.absolute()}/../../images/Kosmos")
    # drawimg_varformat(impath=f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/preprocessed/I_HA_Rep_89_Nr_16160_0090/I_HA_Rep_89_Nr_16160_0090.jpg",predpath=f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/preprocessed/I_HA_Rep_89_Nr_16160_0090", savepath=f"{Path(__file__).parent.absolute()}/../../images/test")
    # drawimg_varformat(savepath=f"{Path(__file__).parent.absolute()}/../../images/Kosmos/tables/test",
    #                  impath=f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/test/IMG_20190821_144501/IMG_20190821_144501.jpg",
    #                  groundpath=f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/test/IMG_20190821_144501",
    #                  predpath=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/BonnData/Tabellen/test/IMG_20190821_144501/IMG_20190821_144501.jpg.json")
    # drawimg_varformat(savepath=f"{Path(__file__).parent.absolute()}/../../images/Kosmos/tables/test",
    #                  impath=f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/test/I_HA_Rep_89_Nr_16160_0213/I_HA_Rep_89_Nr_16160_0213.jpg",
    #                  groundpath=f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/test/I_HA_Rep_89_Nr_16160_0213",
    #                  predpath=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/BonnData/Tabellen/test/I_HA_Rep_89_Nr_16160_0213/I_HA_Rep_89_Nr_16160_0213.jpg.json")

    # drawimg(savepath=f"{Path(__file__).parent.absolute()}/../../images/Kosmos")
    # drawimg(impath= f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/preprocessed/IMG_20190819_115757/IMG_20190819_115757_table_0.jpg", jsonpath=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/IMG_20190819_115757_table_0.jpg.json", savepath=f"{Path(__file__).parent.absolute()}/../../images/Kosmos")

    pass
