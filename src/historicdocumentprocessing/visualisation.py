import glob
import json
import os
from pathlib import Path

import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
from src.historicdocumentprocessing.kosmos_eval import reversetablerelativebboxes_outer, extractboxes


def drawimg_varformat(impath: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData/test/Konflikttabelle.jpg",
                      predpath: str = f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Konflikttabelle.jpg.json",
                      groundpath: str = None,
                      savepath: str = None,
                      tableonly: bool = True):
    #img = plt.imread(impath)
    #img.setflags(write=1)
    #img = img.copy()
    #print(img.flags)
    print(groundpath)
    img = read_image(impath)
    if "json" in predpath:
        with open(predpath) as s:
            box = extractboxes(json.load(s), groundpath if tableonly else None)
            #print(groundpath)
            #print(box)
    elif "pt" in predpath:
        box = torch.load(predpath)
    else:
        box = reversetablerelativebboxes_outer(predpath)

    labels = ["Pred" for i in range(box.shape[0])]
    colors = ["green" for i in range(box.shape[0])]
    #with open(predpath) as s:
    #    box = extractboxes(json.load(s))
    #print(img.shape)
    #print(box, box.numel())
    if not box.numel():
        print(predpath.split("/")[-1])
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
        #print(gbox)
        box = torch.vstack((box, gbox))
    #print(labels, colors, box)
    #print(gbox.shape, box.shape)
    #print(img.shape)
    #print(predpath)
    #print(gbox)
    try:
        image = draw_bounding_boxes(image=img, boxes=box, labels=labels, colors=colors)
    except ValueError:
        #print(predpath)
        #print(box)
        newbox = torch.empty((0, 4))
        for b in box:
            if b[0] < b[2] and b[1] < b[3]:
                newbox= torch.vstack((newbox, b.clone()))
            else:
                print(f"Invalid bounding box in image {impath.split('/')[-1]}",b)
        #print(newbox)
        labels = ["Pred" for i in range(newbox.shape[0])]
        image = draw_bounding_boxes(image=img, boxes=torch.tensor(newbox), labels=labels, colors=colors)
    #plt.imshow(image.permute(1, 2, 0))
    if savepath:
        os.makedirs(savepath, exist_ok=True)
        identifier = impath.split("/")[-1].split(".")[-2]
        #plt.savefig(f"{savepath}/{identifier}")
        #cv2.imwrite(img=image.numpy(), filename=f"{savepath}/{identifier}.jpg")
        img = Image.fromarray(image.permute(1, 2, 0).numpy())
        #print(f"{savepath}/{identifier}.jpg")
        img.save(f"{savepath}/{identifier}.jpg")
    #plt.show()


def drawimg(impath: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData/test/Konflikttabelle.jpg",
            jsonpath: str = f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Konflikttabelle.jpg.json",
            savepath: str = None):
    #img = plt.imread(impath)
    #img.setflags(write=1)
    #img = img.copy()
    #print(img.flags)
    img = read_image(impath)
    with open(jsonpath) as s:
        box = extractboxes(json.load(s))
    #print(img.shape)
    image = draw_bounding_boxes(image=img, boxes=box)
    plt.imshow(image.permute(1, 2, 0))
    if savepath:
        os.makedirs(savepath, exist_ok=True)
        identifier = impath.split("/")[-1].split(".")[-2]
        #plt.savefig(f"{savepath}/{identifier}")
        #cv2.imwrite(img=image.numpy(), filename=f"{savepath}/{identifier}.jpg")
        img = Image.fromarray(image.permute(1, 2, 0).numpy())
        img.save(f"{savepath}/{identifier}.jpg")
    plt.show()


def drawimages(impath: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData/test",
               jsonpath: str = f"{Path(__file__).parent.absolute()}/../../results/kosmos25/BonnData/test",
               savepath: str = None, groundpath: str = None):
    jsons = glob.glob(f"{jsonpath}/*.json")
    for json in tqdm(jsons):
        signifier = json.split("/")[-1].split(".")[-3]
        img = glob.glob(f"{impath}/{signifier}.jpg")
        drawimg_varformat(impath=img[0], predpath=json, savepath=savepath, groundpath=groundpath)


def drawimages_var(impath: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData/test",
                   jsonpath: str = f"{Path(__file__).parent.absolute()}/../../results/kosmos25/BonnData/test",
                   savepath: str = None, groundpath: str = None, tableonly: bool = True, range: int = None):
    jsons = glob.glob(f"{jsonpath}/*.json")
    if len(jsons) == 0:
        jsons = glob.glob(f"{jsonpath}/*")
        #print(len(jsons), jsons)
    for json in tqdm(jsons):
        #print(json)
        if "json" in json:
            signifier = json.split("/")[-1].split(".")[-3]
            #img = glob.glob(f"{impath}/{signifier}.jpg")
            imgpred = json
        else:
            signifier = json.split("/")[-1]
            #print(glob.glob(f"{json}/*"),glob.glob(f"{json}/(?!.*_table_)*$"))
            #imgpred = glob.glob(f"{json}/(?!.*_table_)^.*$")[0]
            imgpred = [file for file in glob.glob(f"{json}/*") if "_table_" not in file][0]
        #signifier = json.split("/")[-1].split(".")[-3]
        #print(signifier, imgpred)
        img = glob.glob(f"{impath}/{signifier}/{signifier}.jpg")
        #print(glob.glob(f"{impath}/{signifier}/*"))
        #print(signifier, img)
        drawimg_varformat(impath=img[0], predpath=imgpred, savepath=savepath,
                          groundpath=groundpath if not groundpath else glob.glob(f"{groundpath}/{signifier}/{signifier}*.pt")[0],
                          tableonly=tableonly)
        if range:
            range-=1
            if range <=0:
                return
        #if groundpath:
        #    groundpathfiner = glob.glob(f"{groundpath}/{signifier}")[0]
        #    #print(glob.glob(groundpathfiner), groundpathfiner)
        #    #print(f"{groundpath}/{signifier}", groundpath)
        #    drawimg_varformat(impath=img[0], predpath=imgpred, savepath=savepath, groundpath=groundpathfiner)
        #else:
        #    drawimg_varformat(impath=img[0], predpath=imgpred, savepath=savepath, groundpath=groundpath)


def main():
    pass


if __name__ == '__main__':
    drawimg_varformat(savepath=f"{Path(__file__).parent.absolute()}/../../images/rcnn/BonnTables/test",
                      impath=f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/test/IMG_20190821_132903/IMG_20190821_132903_table_0.jpg",
                      groundpath=f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/test/IMG_20190821_132903/IMG_20190821_132903_cell_0.pt",
                      predpath=f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/BonnData/IMG_20190821_132903/IMG_20190821_132903.pt")

    #drawimages_var(impath=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/overlaid",
    #                  groundpath=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/overlaid",
    #                  jsonpath=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/overlaid",
    #                  savepath=f"{Path(__file__).parent.absolute()}/../../images/Kosmos/Tablesinthewild/overlaid", tableonly=False, range=50)
    #drawimg_varformat(impath=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/simple/0hZg6EpcTdKXk6j44umj3gAAACMAAQED/0hZg6EpcTdKXk6j44umj3gAAACMAAQED.jpg",
    #                  groundpath=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/simple/0hZg6EpcTdKXk6j44umj3gAAACMAAQED/0hZg6EpcTdKXk6j44umj3gAAACMAAQED.pt",
    #                  predpath=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/simple/0hZg6EpcTdKXk6j44umj3gAAACMAAQED/0hZg6EpcTdKXk6j44umj3gAAACMAAQED.jpg.json",
    #                  savepath=f"{Path(__file__).parent.absolute()}/../../images/Kosmos/Tablesinthewild/simple", tableonly=False)
    #drawimages(savepath=f"{Path(__file__).parent.absolute()}/../../images/Kosmos")
    #drawimages_var()
    #drawimages_var(impath=f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/test",
    #               groundpath=f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/test",
    #               jsonpath=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/BonnData/Tabellen/test",
    #               savepath=f"{Path(__file__).parent.absolute()}/../../images/Kosmos/tables/tableonly")
    #drawimages_var(impath=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
    #               groundpath=f"{Path(__file__).parent.absolute()}/../../data/GloSat/test",
    #               jsonpath=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/GloSat/test1",
    #               savepath=f"{Path(__file__).parent.absolute()}/../../images/Kosmos/GloSat")
    #print(torch.load(f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/preprocessed/I_HA_Rep_89_Nr_16160_0241/I_HA_Rep_89_Nr_16160_0241_tables.pt"))
    #drawimg(impath= f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/preprocessed/I_HA_Rep_89_Nr_16160_0089/I_HA_Rep_89_Nr_16160_0089.jpg", jsonpath=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/I_HA_Rep_89_Nr_16160_0089.jpg.json", savepath=f"{Path(__file__).parent.absolute()}/../../images/Kosmos")
    #drawimg(impath= f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/preprocessed/I_HA_Rep_89_Nr_16160_0089/I_HA_Rep_89_Nr_16160_0089_table_0.jpg",jsonpath=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/I_HA_Rep_89_Nr_16160_0089_table_0.jpg.json",savepath=f"{Path(__file__).parent.absolute()}/../../images/Kosmos")
    #drawimg(jsonpath= f"{Path(__file__).parent.absolute()}/../../results/kosmos25/EngNewspaper/sn83030313_2474-3224_68.jpg.json", impath=f"{Path(__file__).parent.absolute()}/../../data/EngNewspaper/raw/sn83030313_2474-3224_68.jpg", savepath=f"{Path(__file__).parent.absolute()}/../../images/Kosmos")
    #drawimg(jsonpath=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/BonnData/Newspaper/Koelnische_Zeitung_1866-06_1866-09_0046.jpg.json", impath=f"{Path(__file__).parent.absolute()}/../../data/BonnData/Koelnische_Zeitung_1866-06_1866-09_0046.jpg", savepath=f"{Path(__file__).parent.absolute()}/../../images/Kosmos")
    #drawimg_varformat(impath=f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/preprocessed/I_HA_Rep_89_Nr_16160_0090/I_HA_Rep_89_Nr_16160_0090.jpg",predpath=f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/preprocessed/I_HA_Rep_89_Nr_16160_0090", savepath=f"{Path(__file__).parent.absolute()}/../../images/test")
    #drawimg_varformat(savepath=f"{Path(__file__).parent.absolute()}/../../images/Kosmos/tables/test",
    #                  impath=f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/test/IMG_20190821_144501/IMG_20190821_144501.jpg",
    #                  groundpath=f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/test/IMG_20190821_144501",
    #                  predpath=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/BonnData/Tabellen/test/IMG_20190821_144501/IMG_20190821_144501.jpg.json")
    #drawimg_varformat(savepath=f"{Path(__file__).parent.absolute()}/../../images/Kosmos/tables/test",
    #                  impath=f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/test/I_HA_Rep_89_Nr_16160_0213/I_HA_Rep_89_Nr_16160_0213.jpg",
    #                  groundpath=f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/test/I_HA_Rep_89_Nr_16160_0213",
    #                  predpath=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/BonnData/Tabellen/test/I_HA_Rep_89_Nr_16160_0213/I_HA_Rep_89_Nr_16160_0213.jpg.json")

    #drawimg(savepath=f"{Path(__file__).parent.absolute()}/../../images/Kosmos")
    #drawimg(impath= f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/preprocessed/IMG_20190819_115757/IMG_20190819_115757_table_0.jpg", jsonpath=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/IMG_20190819_115757_table_0.jpg.json", savepath=f"{Path(__file__).parent.absolute()}/../../images/Kosmos")

    pass
