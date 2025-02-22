import glob
import os
from pathlib import Path
from typing import List, Literal

import torch
from lightning_fabric.utilities import move_data_to_device
from matplotlib import pyplot as plt
from torch.nn.functional import threshold
from torchvision.io import read_image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from tqdm import tqdm
from transformers import TableTransformerForObjectDetection

from src.historicdocumentprocessing.kosmos_eval import (
    calcstats_IoU,
    reversetablerelativebboxes_outer,
)
from src.historicdocumentprocessing.tabletransformer_dataset import CustomDataset
from src.historicdocumentprocessing.util.plottotikz import save_plot_as_tikz
from src.historicdocumentprocessing.util.tablesutil import getcells


def findoptimalfilterpoint_outer(
    modellist: List[str],
    modelfolder: str = f"{Path(__file__).parent.absolute()}/../../../checkpoints/fasterrcnn",
    valid: bool = True,
    modeltype: Literal["fasterrcnn", "tabletransformer"] = "fasterrcnn",
):
    # for modelpath in glob.glob(f"{modelfolder}/*"):
    for modelname in modellist:
        modelpath = f"{modelfolder}/{modelname}"
        if "BonnData" in modelpath and "run" not in modelpath:
            if modeltype == "fasterrcnn":
                findoptimalfilterpoint(
                    modelpath,
                    testdatasetpath=(
                        f"{Path(__file__).parent.absolute()}/../../../data/BonnData/test"
                        if not valid
                        else f"{Path(__file__).parent.absolute()}/../../../data/BonnData/valid"
                    ),
                    valid=valid,
                )
            else:
                findoptimalfilterpoint_tabletransformer(
                    modelpath,
                    datasetpath=f"{Path(__file__).parent.absolute()}/../../../data/BonnData",
                    valid=valid,
                )
        elif (
            "GloSat" in modelpath
            and "BonnData" not in modelpath
            and "run" not in modelpath
        ):
            if modeltype == "fasterrcnn":
                findoptimalfilterpoint(
                    modelpath,
                    testdatasetpath=(
                        f"{Path(__file__).parent.absolute()}/../../../data/GloSat/test"
                        if not valid
                        else f"{Path(__file__).parent.absolute()}/../../../data/GloSat/valid"
                    ),
                    valid=valid,
                )
            else:
                findoptimalfilterpoint_tabletransformer(
                    modelpath,
                    datasetpath=f"{Path(__file__).parent.absolute()}/../../../data/GloSat",
                    valid=valid,
                )
        elif "Tablesinthewild" or "titw" in modelpath:
            if modeltype == "fasterrcnn":
                findoptimalfilterpoint(
                    modelpath,
                    testdatasetpath=(
                        f"{Path(__file__).parent.absolute()}/../../../data/Tablesinthewild/test"
                        if not valid
                        else f"{Path(__file__).parent.absolute()}/../../../data/Tablesinthewild/valid"
                    ),
                    tablerelative=False,
                    valid=valid,
                )
            else:
                findoptimalfilterpoint_tabletransformer(
                    modelpath,
                    datasetpath=f"{Path(__file__).parent.absolute()}/../../../data/Tablesinthewild",
                    valid=valid,
                )
        else:
            pass


def findoptimalfilterpoint_tabletransformer(
    modelpath: str, datasetpath: str, valid: bool = True
):
    """calculate best filterpoint for tabletransformer (based on row/column iou)"""
    modelname = "base_table-transformer-structure-recognition"
    model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-structure-recognition"
    )
    if modelpath:
        modelname = modelpath.split("/")[-1]
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
    dataset = CustomDataset(
        f"{datasetpath}/{'valid' if valid else 'test'}",
        "fullimage",
        transforms=None,
    )
    totalscores = []
    totaliou = []
    for i in tqdm(range(len(dataset))):
        print(dataset.getfolder(i))
        img, fullimagegroundbox, labels = dataset.getimgtarget(i, addtables=False)
        encoding = move_data_to_device(
            dataset.ImageProcessor(img, return_tensors="pt"), device=device
        )
        folder = dataset.getfolder(i)
        imname = folder.split("/")[-1]
        with torch.no_grad():
            results = model(**encoding)
        width, height = img.size
        output = dataset.ImageProcessor.post_process_object_detection(
            results, threshold=0.0, target_sizes=[(height, width)]
        )[0]
        # print(output)
        # labels = torch.Tensor([str(model.config.id2label[label]) for label in output["labels"].tolist()])
        boxes = torch.vstack(
            [
                output["boxes"][output["labels"] == 2],
                output["boxes"][output["labels"] == 1],
            ]
        ).to(device="cpu")
        predious, _, tp, fp, _ = calcstats_IoU(
            predbox=boxes, targetbox=fullimagegroundbox
        )
        predscores = list(
            torch.hstack(
                [
                    output["scores"][output["labels"] == 2],
                    output["scores"][output["labels"] == 1],
                ]
            ).to(device="cpu")
        )
        # print(predscores)
        assert len(predscores) == len(predious)
        totalscores += predscores
        totaliou += predious
    # print(totalscores)
    # print(totaliou)
    # print(output)
    bestpred, preds, sum_fp, sum_tp = findoptimalfilterpoint_inner(
        totaliou, totalscores
    )
    fig = plt.figure()
    plt.title(
        "true positives and false positives at iou_threshold 0.5 by bbox probability score"
    )
    plt.plot(preds, sum_tp, color="green", label="tp")
    plt.plot(preds, sum_fp, color="red", label="fp")
    plt.xlabel("bbox probability score")
    plt.ylabel("number of tp/fp")
    plt.legend()
    os.makedirs(
        f"{Path(__file__).parent.absolute()}/../../../images/tabletransformer/{modelname}",
        exist_ok=True,
    )
    plt.savefig(
        f"{Path(__file__).parent.absolute()}/../../../images/tabletransformer/{modelname}/threshold_graph{'_valid' if valid else ''}.png"
    )
    os.makedirs(
        f"{Path(__file__).parent.absolute()}/../../../tikzplots/tabletransformer/{modelname}",
        exist_ok=True,
    )
    save_plot_as_tikz(
        fig=fig,
        savepath=f"{Path(__file__).parent.absolute()}/../../../tikzplots/tabletransformer/{modelname}/threshold_graph{'_valid' if valid else ''}.tex",
    )
    os.makedirs(
        f"{Path(__file__).parent.absolute()}/../../../results/tabletransformer/bestfilterthresholds{'_valid' if valid else ''}",
        exist_ok=True,
    )
    plt.close()
    with open(
        f"{Path(__file__).parent.absolute()}/../../../results/tabletransformer/bestfilterthresholds{'_valid' if valid else ''}/{modelname}.txt",
        "w",
    ) as f:
        f.write(str(bestpred))
    pass


def findoptimalfilterpoint(
    modelpath: str, testdatasetpath: str, tablerelative: bool = True, valid: bool = True
):
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
    modelname = modelpath.split("/")[-1]
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        model.eval()
    else:
        print("Cuda not available")
        return
    totaliou = []
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
        predious, _, tp, fp, _ = calcstats_IoU(
            predbox=output["boxes"], targetbox=fullimagegroundbox
        )
        predscores = list(output["scores"])
        assert len(predscores) == len(predious)
        totalscores += predscores
        totaliou += predious
        # print(predscores)
    print(totaliou)
    print(totalscores)
    bestpred, preds, sum_fp, sum_tp = findoptimalfilterpoint_inner(
        totaliou, totalscores
    )
    # print(sum_tp/sum_fp)
    # print(torch.argmax(sum_tp-sum_fp))
    fig = plt.figure()
    plt.title("tp/fp @ IoU 0.5 by bbox probability score")
    plt.plot(preds, sum_tp, color="green", label="tp")
    plt.plot(preds, sum_fp, color="red", label="fp")
    plt.xlabel("bbox probability score")
    plt.ylabel("number of tp/fp")
    plt.legend()
    os.makedirs(
        f"{Path(__file__).parent.absolute()}/../../../images/fasterrcnn/{modelname}",
        exist_ok=True,
    )
    plt.savefig(
        f"{Path(__file__).parent.absolute()}/../../../images/fasterrcnn/{modelname}/threshold_graph{'_valid' if valid else ''}.png"
    )
    os.makedirs(
        f"{Path(__file__).parent.absolute()}/../../../tikzplots/fasterrcnn/{modelname}",
        exist_ok=True,
    )
    save_plot_as_tikz(
        fig=fig,
        savepath=f"{Path(__file__).parent.absolute()}/../../../tikzplots/fasterrcnn/{modelname}/threshold_graph{'_valid' if valid else ''}.tex",
    )
    os.makedirs(
        f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/bestfilterthresholds{'_valid' if valid else ''}",
        exist_ok=True,
    )
    plt.close()
    with open(
        f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/bestfilterthresholds{'_valid' if valid else ''}/{modelname}.txt",
        "w",
    ) as f:
        f.write(str(bestpred))
    # print(bestpred)


def findoptimalfilterpoint_inner(totaliou, totalscores):
    bestpred = 0
    totalscores = torch.tensor(totalscores)
    totaliou = torch.tensor(totaliou)
    preds, _ = torch.sort(torch.unique(totalscores))
    sum_tp = torch.zeros((len(preds)))
    sum_fp = torch.zeros((len(preds)))
    # print(preds)
    # print(totalscores)
    # print(totaliou)
    for idx, pred in enumerate(preds):
        assert torch.equal(
            len(totaliou[totalscores >= pred])
            - torch.sum(totaliou[totalscores >= pred] >= 0.5),
            torch.sum(totaliou[totalscores >= pred] < 0.5),
        )
        # bestpred += pred * (torch.sum(totaliou[totalscores == pred] >= 0.5) / totaliou[totalscores == pred])
        sum_tp[idx] = torch.sum(totaliou[totalscores >= pred] >= 0.5)
        sum_fp[idx] = torch.sum(totaliou[totalscores >= pred] < 0.5)
    bestpred = round(torch.min(preds[torch.argmax(sum_tp - sum_fp)]).item(), 2)
    ###edge cases#######
    if bestpred == 0.0:
        bestpred = 0.01
    elif bestpred == 1.0:
        bestpred = 0.99
    return bestpred, preds, sum_fp, sum_tp


if __name__ == "__main__":
    # findoptimalfilterpoint_outer(valid=True, modellist=['GloSatFullImage1_GloSat_fullimage_e250_es.pt',
    #                                                                 'GloSatFullImage_random_GloSat_fullimage_e250_random_init_es.pt','BonnDataFullImage1_BonnData_fullimage_e250_es.pt',
    #                                                                 'BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_es.pt', 'testseveralcalls_4_with_valid_split_Tablesinthewild_fullimage_e50_es.pt',
    #                              'testseveralcalls_valid_random_init_e_250_es.pt'])
    # findoptimalfilterpoint_outer(valid=False,modellist=['GloSatFullImage1_GloSat_fullimage_e250_es.pt',
    #                                                                 'GloSatFullImage_random_GloSat_fullimage_e250_random_init_es.pt','BonnDataFullImage1_BonnData_fullimage_e250_es.pt',
    #                                                                 'BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_es.pt', 'testseveralcalls_4_with_valid_split_Tablesinthewild_fullimage_e50_es.pt',
    #                              'testseveralcalls_valid_random_init_e_250_es.pt'])

    findoptimalfilterpoint_outer(
        modellist=[
            # "tabletransformer_v0_new_BonnDataFullImage_tabletransformer_estest_BonnData_fullimage_e1_init_tabletransformer_v0_new_GloSatFullImage_tabletransformer_newenv_fixed_GloSat_fullimage_e250_valid_es.pt_valid_es.pt", "tabletransformer_v0_new_BonnDataFullImage_tabletransformer_estest_BonnData_fullimage_e1_init_tabletransformer_v0_new_GloSatFullImage_tabletransformer_newenv_fixed_GloSat_fullimage_e250_valid_es.pt_valid_end.pt", "tabletransformer_v0_new_BonnDataFullImage_tabletransformer_estest_BonnData_fullimage_e250_valid_es.pt", "tabletransformer_v0_new_BonnDataFullImage_tabletransformer_estest_BonnData_fullimage_e250_valid_end.pt", "tabletransformer_v0_new_GloSatFullImage_tabletransformer_newenv_fixed_GloSat_fullimage_e250_valid_es.pt", "tabletransformer_v0_new_GloSatFullImage_tabletransformer_newenv_fixed_GloSat_fullimage_e250_valid_end.pt", "tabletransformer_v0_new_BonnDataFullImage_tabletransformer_loadtest_BonnData_fullimage_e250_init_tabletransformer_v0_new_GloSatFullImage_tabletransformer_newenv_fixed_GloSat_fullimage_e250_valid_es.pt_valid_es.pt",
            "titw_severalcalls_2_e250_es.pt",
            "titw_severalcalls_2_e250_end.pt",
            "titw_call_aachen_e250_es.pt",
            "titw_call_aachen_e250_end.pt",
        ],
        modelfolder=f"{Path(__file__).parent.absolute()}/../../../checkpoints/tabletransformer",
        valid=True,
        modeltype="tabletransformer",
    )
    # findoptimalfilterpoint_outer(modellist=[
    # "tabletransformer_v0_new_BonnDataFullImage_tabletransformer_estest_BonnData_fullimage_e1_init_tabletransformer_v0_new_GloSatFullImage_tabletransformer_newenv_fixed_GloSat_fullimage_e250_valid_es.pt_valid_es.pt",
    # "tabletransformer_v0_new_BonnDataFullImage_tabletransformer_estest_BonnData_fullimage_e1_init_tabletransformer_v0_new_GloSatFullImage_tabletransformer_newenv_fixed_GloSat_fullimage_e250_valid_es.pt_valid_end.pt",
    # "tabletransformer_v0_new_BonnDataFullImage_tabletransformer_estest_BonnData_fullimage_e250_valid_es.pt",
    # "tabletransformer_v0_new_BonnDataFullImage_tabletransformer_estest_BonnData_fullimage_e250_valid_end.pt",
    # "tabletransformer_v0_new_GloSatFullImage_tabletransformer_newenv_fixed_GloSat_fullimage_e250_valid_es.pt",
    # "tabletransformer_v0_new_GloSatFullImage_tabletransformer_newenv_fixed_GloSat_fullimage_e250_valid_end.pt", "tabletransformer_v0_new_BonnDataFullImage_tabletransformer_loadtest_BonnData_fullimage_e250_init_tabletransformer_v0_new_GloSatFullImage_tabletransformer_newenv_fixed_GloSat_fullimage_e250_valid_es.pt_valid_es.pt",
    #    "titw_severalcalls_2_e250_es.pt", "titw_severalcalls_2_e250_end.pt", "titw_call_aachen_e250_es.pt", "titw_call_aachen_e250_end.pt"],
    #                             modelfolder=f"{Path(__file__).parent.absolute()}/../../../checkpoints/tabletransformer",
    #                             valid=False, modeltype="tabletransformer")
    # findoptimalfilterpoint(modelpath=f"{Path(__file__).parent.absolute()}/../../../checkpoints/fasterrcnn/BonnDataFullImage1_BonnData_fullimage_e250_es.pt", testdatasetpath=f"{Path(__file__).parent.absolute()}/../../../data/BonnData/test", valid=True)
