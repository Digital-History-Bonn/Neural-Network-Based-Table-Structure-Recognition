"""Visualise predictions on image."""
import argparse
import glob
import json
from pathlib import Path
from typing import Optional, List

import torch
from lightning_fabric.utilities import move_data_to_device
from PIL import Image
from torchvision.io import read_image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from tqdm import tqdm
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

from src.historicdocumentprocessing.postprocessing import postprocess
from src.historicdocumentprocessing.util.tablesutil import getcells, remove_invalid_bbox, \
    reversetablerelativebboxes_outer, extractboxes
from src.historicdocumentprocessing.util.visualisationutil import (
    drawimg_varformat_inner,
)


def drawimg_allmodels(
    datasetname: str,
    imgpath: str,
    rcnnmodels: Optional[List[str]] = None,
    tabletransformermodels: Optional[List[str]] = None,
    predfolder: Optional[str] = None,
    valid: bool = True,
):
    """Method to draw image for all models.

    Args:
        datasetname: name of dataset
        imgpath: path to image
        rcnnmodels: list of RCNN model names
        tabletransformermodels: list of Tabletransformer model names
        predfolder: folder of kosmos predictions (if any)
        valid: wether to use valid filter thresholds

    """
    if rcnnmodels is None:
        rcnnmodels = []
    if tabletransformermodels is None:
        tabletransformermodels = []
    imname = imgpath.split("/")[-1].split(".")[-2]
    groundpath = (
        f"{Path(__file__).parent.absolute()}/../../data/{datasetname.split('/')[-2]}/preprocessed/{datasetname.split('/')[-1]}"
        if "/" in datasetname
        else f"{Path(__file__).parent.absolute()}/../../data/{datasetname}/test"
    )
    groundpath = f"{groundpath}/{imname}"
    predpath_kosmos = f'{f"{Path(__file__).parent.absolute()}/../../results/kosmos25/{datasetname}/{imname}/{imname}" if not predfolder else f"{Path(__file__).parent.absolute()}/../../results/kosmos25/{datasetname}/{predfolder}/{imname}/{imname}"}.jpg.json'
    predpath_kosmos_postprocessed = f"{Path(__file__).parent.absolute()}/../../results/kosmos25/postprocessed/fullimg/{datasetname}/eps_3.0_1.5/withoutoutlier_excludeoutliers_glosat/{imname}/{imname}.pt"
    predpath_rcnn_postprocessed = f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/postprocessed/fullimg/{datasetname}"
    for name, path in zip(
        ["not_postprocessed", "postprocessed"],
        [predpath_kosmos, predpath_kosmos_postprocessed],
    ):
        savepath = f"{Path(__file__).parent.absolute()}/../../images/kosmos25/{name}"
        drawimg_varformat(
            impath=imgpath,
            predpath=path,
            groundpath=groundpath,
            tableonly=False,
            savepath=savepath,
        )
    for model in glob.glob(f"{predpath_rcnn_postprocessed}/*"):
        if "end" not in model:
            savepath = f"{Path(__file__).parent.absolute()}/../../images/fasterrcnn/{model.split('/')[-1]}/postprocessed"
            try:
                drawimg_varformat(
                    impath=imgpath,
                    predpath=f"{model}/eps_3.0_1.5/withoutoutlier_excludeoutliers_glosat/{imname}/{imname}.pt",
                    groundpath=groundpath,
                    tableonly=False,
                    savepath=savepath,
                )
            except FileNotFoundError:
                print(f"{imname} has no valid predicitons with model {model}")
                pass
    for modelname in rcnnmodels:
        modelpath = f"{Path(__file__).parent.absolute()}/../../checkpoints/fasterrcnn/{modelname}"
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
        try:
            with open(
                f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/bestfilterthresholds{'_valid' if valid else ''}/{modelname}.txt",
                "r",
            ) as f:
                filter = float(f.read())
        except FileNotFoundError:
            filter = 0.7
        fullimagepredbox_filtered = remove_invalid_bbox(
            output["boxes"][output["scores"] > filter]
        )
        fullimagepredbox = remove_invalid_bbox(output["boxes"])
        # print("imgpath", imgpath)
        drawimg_varformat_inner(
            impath=imgpath,
            box=fullimagepredbox,
            groundpath=groundpath,
            savepath=f"{Path(__file__).parent.absolute()}/../../images/fasterrcnn/{modelpath.split('/')[-1]}",
        )
        drawimg_varformat_inner(
            impath=imgpath,
            box=fullimagepredbox_filtered,
            groundpath=groundpath,
            savepath=f"{Path(__file__).parent.absolute()}/../../images/fasterrcnn/{modelpath.split('/')[-1]}/filtered_{filter}{'_valid' if valid else ''}",
        )
        filtered_processed = postprocess(
            fullimagepredbox_filtered,
            imname=imname,
            saveloc=f"{Path(__file__).parent.absolute()}/../../images/test.pt",
            minsamples=[2, 3],
        )
        drawimg_varformat_inner(
            impath=imgpath,
            box=filtered_processed,
            groundpath=groundpath,
            savepath=f"{Path(__file__).parent.absolute()}/../../images/fasterrcnn/{modelpath.split('/')[-1]}/filtered_{filter}{'_valid' if valid else ''}_postprocessed",
        )
    for modelname in tabletransformermodels:
        model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        )
        modelpath = f"{Path(__file__).parent.absolute()}/../../checkpoints/tabletransformer/{modelname}"
        # model.load_state_dict(torch.load(modelpath))
        if "call" in modelname:
            model.load_state_dict(torch.load(modelpath, map_location="cuda:0"))
        else:
            model.load_state_dict(torch.load(modelpath))
        print("loaded_model:", modelname)
        assert model.config.id2label[1] == "table column"
        assert model.config.id2label[2] == "table row"
        # image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        # image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model.to(device)
            model.eval()
        else:
            print("Cuda not available")
            return
        try:
            with open(
                f"{Path(__file__).parent.absolute()}/../../results/tabletransformer/bestfilterthresholds{'_valid' if valid else ''}/{modelname}.txt",
                "r",
            ) as f:
                filterthreshold = float(f.read())
                if filterthreshold == 1.0:
                    filterthreshold = 0.99
                if filterthreshold == 0.0:
                    filterthreshold = 0.01
        except FileNotFoundError:
            filterthreshold = 0.7
        # print(filterthreshold)
        img = Image.open(imgpath).convert("RGB")
        image_processor = AutoImageProcessor.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        )
        encoding = move_data_to_device(
            image_processor(img, return_tensors="pt"), device=device
        )
        with torch.no_grad():
            results = model(**encoding)
        width, height = img.size
        output = image_processor.post_process_object_detection(
            results, threshold=0.0, target_sizes=[(height, width)]
        )[0]
        cellboxes = getcells(
            rows=output["boxes"][output["labels"] == 2],
            cols=output["boxes"][output["labels"] == 1],
        ).to(device="cpu")
        boxes = torch.vstack(
            [
                output["boxes"][output["labels"] == 2],
                output["boxes"][output["labels"] == 1],
            ]
        ).to(device="cpu")
        outputfiltered = image_processor.post_process_object_detection(
            results, threshold=filterthreshold, target_sizes=[(height, width)]
        )[0]
        cellboxesfiltered = getcells(
            rows=outputfiltered["boxes"][outputfiltered["labels"] == 2],
            cols=outputfiltered["boxes"][outputfiltered["labels"] == 1],
        ).to(device="cpu")
        boxesfiltered = torch.vstack(
            [
                outputfiltered["boxes"][outputfiltered["labels"] == 2],
                outputfiltered["boxes"][outputfiltered["labels"] == 1],
            ]
        ).to(device="cpu")
        drawimg_varformat_inner(
            impath=imgpath,
            box=cellboxes,
            groundpath=groundpath,
            savepath=f"{Path(__file__).parent.absolute()}/../../images/tabletransformer/{modelpath.split('/')[-1]}/cellimg",
        )
        drawimg_varformat_inner(
            impath=imgpath,
            box=cellboxesfiltered,
            groundpath=groundpath,
            savepath=f"{Path(__file__).parent.absolute()}/../../images/tabletransformer/{modelpath.split('/')[-1]}/cellimg_filtered_{filterthreshold}{'_valid' if valid else ''}",
        )
        drawimg_varformat_inner(
            impath=imgpath,
            box=boxes,
            groundpath=groundpath,
            rowcol=True,
            savepath=f"{Path(__file__).parent.absolute()}/../../images/tabletransformer/{modelpath.split('/')[-1]}/rowcolimg",
        )
        drawimg_varformat_inner(
            impath=imgpath,
            box=boxesfiltered,
            groundpath=groundpath,
            rowcol=True,
            savepath=f"{Path(__file__).parent.absolute()}/../../images/tabletransformer/{modelpath.split('/')[-1]}/rowcolimg_filtered_{filterthreshold}{'_valid' if valid else ''}",
        )
        postprocessed = postprocess(
            cellboxes,
            imname=imname,
            saveloc=f"{Path(__file__).parent.absolute()}/../../images/test.pt",
            minsamples=[2, 3],
        )
        postprocessed_filtered = postprocess(
            cellboxesfiltered,
            imname=imname,
            saveloc=f"{Path(__file__).parent.absolute()}/../../images/test.pt",
            minsamples=[2, 3],
        )
        drawimg_varformat_inner(
            impath=imgpath,
            box=postprocessed_filtered,
            groundpath=groundpath,
            savepath=f"{Path(__file__).parent.absolute()}/../../images/tabletransformer/{modelpath.split('/')[-1]}/cellimg_filtered_{filterthreshold}{'_valid' if valid else ''}_postprocessed",
        )
        drawimg_varformat_inner(
            impath=imgpath,
            box=postprocessed,
            groundpath=groundpath,
            savepath=f"{Path(__file__).parent.absolute()}/../../images/tabletransformer/{modelpath.split('/')[-1]}/cellimg_postprocessed",
        )


def drawimg_varformat(
    impath: str,  # f"{Path(__file__).parent.absolute()}/../../data/BonnData/test/Konflikttabelle.jpg"
    predpath: str,  # f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Konflikttabelle.jpg.json"
    groundpath: Optional[str] = None,
    savepath: Optional[str] = None,
    tableonly: bool = True,
):
    """Draw prediction (and ground truth) on image (when prediciton has been saved to file).

    Args:
        impath: image path
        predpath: prediction path
        groundpath: ground truth path
        savepath: where to save image
        tableonly: wether to draw all predictions or only predictions in table area

    """
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
    if not box.numel():
        print(predpath.split("/")[-1])

    drawimg_varformat_inner(
        box=box, groundpath=groundpath, impath=impath, savepath=savepath
    )


def drawimages(
    impath: str,  # f"{Path(__file__).parent.absolute()}/../../data/BonnData/test"
    jsonpath: str,  # f"{Path(__file__).parent.absolute()}/../../results/kosmos25/BonnData/test"
    savepath: Optional[str] = None,
    groundpath: Optional[str] = None,
):
    """Draw images with prediction and groundtruth for all json files in a folder (kosmos).

    Args:
        impath: path to image folder
        jsonpath: path to json files
        savepath: where to save image
        groundpath: path to ground truth files

    """
    jsons = glob.glob(f"{jsonpath}/*.json")
    for j in tqdm(jsons):
        signifier = j.split("/")[-1].split(".")[-3]
        img = glob.glob(f"{impath}/{signifier}.jpg")
        drawimg_varformat(
            impath=img[0], predpath=j, savepath=savepath, groundpath=groundpath
        )


def drawimages_var(
    impath: str,  # f"{Path(__file__).parent.absolute()}/../../data/BonnData/test"
    jsonpath: str,  # f"{Path(__file__).parent.absolute()}/../../results/kosmos25/BonnData/test"
    savepath: Optional[str] = None,
    groundpath: Optional[str] = None,
    tableonly: bool = True,
    range: Optional[int] = None,
):
    """Draw all images in folder of saved json/pt predictions.

    Args:
        impath: image folder
        jsonpath: path to json/pt files or subfolders
        savepath: where to save
        groundpath: path to groundtruth files
        tableonly:  wether to draw all predictions or only predictions in table area
        range: max number of images to draw

    """
    jsons = glob.glob(f"{jsonpath}/*.json")
    if len(jsons) == 0:
        jsons = glob.glob(f"{jsonpath}/*")
    for j in tqdm(jsons):
        if "json" in j:
            signifier = j.split("/")[-1].split(".")[-3]
            imgpred = j
        else:
            signifier = j.split("/")[-1]
            imgpred = [
                file for file in glob.glob(f"{j}/*") if "_table_" not in file
            ][0]
        img = glob.glob(f"{impath}/{signifier}/{signifier}.jpg")
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


def get_args() -> argparse.Namespace:
    """Define args."""   # noqa: DAR201
    parser = argparse.ArgumentParser(description="visualisation")
    parser.add_argument('i', '--imname', default="test", help="name of image")
    parser.add_argument('-p', '--predfolder', default='', help="prediction folder or folders")
    parser.add_argument('--datasetname', default="BonnData")
    parser.add_argument('--rcnnmodels', nargs='*', type=str, default=[])
    parser.add_argument('--tabletransformermodels', nargs='*', type=str, default=[])

    parser.add_argument('--valid', action='store_true', default=False)
    parser.add_argument('--no-valid', dest='tableareaonly', action='store_false')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    impath = f"{Path(__file__).parent.absolute()}/../../data/{args.datasetname}/{args.predfolder}/{args.imname}/{args.imname}.pt"
    drawimg_allmodels(datasetname=args.datasetname, imgpath=impath, rcnnmodels=args.rcnnmodels, tabletransformermodels=args.tabletransformermodels, predfolder=args.predfolder, valid=args.valid)
    exit()

    # drawimg_varformat(savepath=f"{Path(__file__).parent.absolute()}/../../images/rcnn/BonnTables/test",
    #                  impath=f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/test/IMG_20190821_132903/IMG_20190821_132903_table_0.jpg",
    #                  groundpath=f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/test/IMG_20190821_132903/IMG_20190821_132903_cell_0.pt",
    #                  predpath=f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/BonnData/IMG_20190821_132903/IMG_20190821_132903.pt")
    # drawimg_varformat(
    #    impath=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/curved/table_spider_00909/table_spider_00909.jpg",
    #    groundpath=f"{Path(__file__).parent.absolute()}/../../data/Tablesinthewild/preprocessed/curved/table_spider_00909/table_spider_00909.pt",
    #    predpath=f"{Path(__file__).parent.absolute()}/../../results/kosmos25/Tablesinthewild/curved/table_spider_00909/table_spider_00909.jpg.json",
    #    savepath=f"{Path(__file__).parent.absolute()}/../../images/test/tablespider",
    #    tableonly=False,
    # )
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
