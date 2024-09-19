import glob
import os
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from torchvision.io import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from tqdm import tqdm

from src.historicdocumentprocessing.kosmos_eval import calcstats_IoU, reversetablerelativebboxes_outer

def findoptimalfilterpoint_outer(modelfolder:str= f"{Path(__file__).parent.absolute()}/../../../checkpoints/fasterrcnn", valid: bool = True):
    for modelpath in glob.glob(f"{modelfolder}/*"):
        if "BonnData" in modelpath and "run" not in modelpath:
            findoptimalfilterpoint(modelpath, testdatasetpath=f"{Path(__file__).parent.absolute()}/../../../data/BonnData/test" if not valid else f"{Path(__file__).parent.absolute()}/../../../data/BonnData/valid", valid=valid)
        elif "GloSat" in modelpath and "BonnData" not in modelpath and "run" not in modelpath:
            findoptimalfilterpoint(modelpath,
                                   testdatasetpath=f"{Path(__file__).parent.absolute()}/../../../data/GloSat/test" if not valid else f"{Path(__file__).parent.absolute()}/../../../data/GloSat/valid", valid=valid)
        elif "Tablesinthewild" in modelpath:
            findoptimalfilterpoint(modelpath,
                                   testdatasetpath=f"{Path(__file__).parent.absolute()}/../../../data/Tablesinthewild/test" if not valid else f"{Path(__file__).parent.absolute()}/../../../data/Tablesinthewild/valid", tablerelative=False, valid=valid)
        else:
            pass

def findoptimalfilterpoint(modelpath:str, testdatasetpath:str, tablerelative:bool= True, valid:bool=True):
    """
    partially based on threshold graph function from https://github.com/Digital-History-Bonn/HistorischeTabellenSemanticExtraction/blob/main/src/TableExtraction/utils/metrics.py
    Args:
        modelpath:
        testdatasetpath:
        tablerelative:

    Returns:

    """
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        **{"box_detections_per_img": 200},
    )
    model.load_state_dict(torch.load(modelpath))
    modelname = modelpath.split('/')[-1]
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        model.eval()
    else:
        print("Cuda not available")
        return
    totaliou= []
    totalscores = []
    for n, folder in tqdm(enumerate(glob.glob(f"{testdatasetpath}/*"))):
        # print(folder)
        impath = f"{folder}/{folder.split('/')[-1]}.jpg"
        imname = folder.split("/")[-1]
        img = (read_image(impath) / 255).to(device)
        if tablerelative:
            fullimagegroundbox = reversetablerelativebboxes_outer(folder)
        else:
            fullimagegroundbox = torch.load(
                glob.glob(f"{folder}/{folder.split('/')[-1]}.pt")[0]
            )
        output = model([img])
        output = {k: v.detach().cpu() for k, v in output[0].items()}
        predious,_,tp,fp,_ = calcstats_IoU(predbox=output['boxes'], targetbox=fullimagegroundbox)
        predscores = list(output['scores'])
        assert len(predscores)==len(predious)
        totalscores+=predscores
        totaliou+=predious
        #print(predscores)
    bestpred =0
    totalscores = torch.tensor(totalscores)
    totaliou = torch.tensor(totaliou)
    preds,_ = torch.sort(torch.unique(totalscores))
    sum_tp = torch.zeros((len(preds)))
    sum_fp = torch.zeros((len(preds)))
    #print(preds)
    #print(totalscores)
    #print(totaliou)
    for idx,pred in enumerate(preds):
        assert torch.equal(len(totaliou[totalscores>=pred])-torch.sum(totaliou[totalscores>=pred]>=0.5),torch.sum(totaliou[totalscores>=pred]<0.5))
        #bestpred += pred * (torch.sum(totaliou[totalscores == pred] >= 0.5) / totaliou[totalscores == pred])
        sum_tp[idx]= torch.sum(totaliou[totalscores>=pred]>=0.5)
        sum_fp[idx]= torch.sum(totaliou[totalscores>=pred]<0.5)
    bestpred = round(torch.min(preds[torch.argmax(sum_tp-sum_fp)]).item(),2)
    #print(sum_tp/sum_fp)
    #print(torch.argmax(sum_tp-sum_fp))
    plt.title('true positives and false positives at iou_threshold 0.5 by bbox probability score')
    plt.plot(preds, sum_tp, color='green', label='tp')
    plt.plot(preds, sum_fp, color='red', label='fp')
    plt.xlabel('bbox probability score')
    plt.ylabel('number of tp/fp')
    plt.legend()
    os.makedirs(f"{Path(__file__).parent.absolute()}/../../../images/fasterrcnn/{modelname}", exist_ok=True)
    plt.savefig(f"{Path(__file__).parent.absolute()}/../../../images/fasterrcnn/{modelname}/threshold_graph{'_valid' if valid else ''}.png")
    os.makedirs(f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/bestfilterthresholds{'_valid' if valid else ''}", exist_ok=True)
    plt.close()
    with open(f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/bestfilterthresholds{'_valid' if valid else ''}/{modelname}.txt", 'w') as f:
        f.write(str(bestpred))
    #print(bestpred)


if __name__=='__main__':
    findoptimalfilterpoint_outer(valid=True)
    #findoptimalfilterpoint(modelpath=f"{Path(__file__).parent.absolute()}/../../../checkpoints/fasterrcnn/BonnDataFullImage1_BonnData_fullimage_e250_es.pt", testdatasetpath=f"{Path(__file__).parent.absolute()}/../../../data/BonnData/test")
