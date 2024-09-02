import glob
import os
from pathlib import Path

import pandas
import torch
from PIL import Image
from tqdm import tqdm
from  transformers import AutoModelForObjectDetection, DetrImageProcessor, TableTransformerForObjectDetection

from src.historicdocumentprocessing.kosmos_eval import reversetablerelativebboxes_outer
from src.historicdocumentprocessing.tabletransformer_dataset import CustomDataset


def inference(modelpath:str=None, targetloc=None, tablerelative=None, iou_thresholds=None, tableareaonly=None,
              filtering=None):
    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition" if not modelpath else modelpath)
    image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        model.eval()
    else:
        print("Cuda not available")
        return

    saveloc = f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/testevalfinal1/fullimg/{datasetname}/{modelname}/iou_{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}"
    # boxsaveloc = f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/{datasetname}/{modelname}"
    if tableareaonly and not filtering:
        saveloc = f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/testevalfinal1/fullimg/{datasetname}/{modelname}/tableareaonly/iou_{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}"
    elif filtering and not tableareaonly:
        saveloc = f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/testevalfinal1/fullimg/{datasetname}/{modelname}/filtering_iou{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}_scorethreshold_0.7"
    elif filtering and tableareaonly:
        saveloc = f"{Path(__file__).parent.absolute()}/../../results/fasterrcnn/testevalfinal1/fullimg/{datasetname}/{modelname}/tableareaonly/filtering_iou{'_'.join([str(iou_thresholds[0]), str(iou_thresholds[-1])])}_scorethreshold_0.7"
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
    for i in len(dataset):
        img, target = dataset.getimgtarget(i)
        encoding = dataset[i]
        results = model(**encoding)
        width, height = img.size
        if filtering:
            output = dataset.ImageProcessor.post_process_object_detection(results, threshold=0.7, target_sizes=[(height, width)])[0]
        else:
            output = dataset.ImageProcessor.post_process_object_detection(results, target_sizes=[(height, width)])[0]
        if tableareaonly:
            output["boxes"] = tableareabboxes(output["boxes"], folder)
        fullimagepredbox = output["boxes"]

