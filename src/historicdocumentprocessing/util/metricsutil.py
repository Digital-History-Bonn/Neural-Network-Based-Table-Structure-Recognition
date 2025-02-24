import glob
import os
import warnings
from pathlib import Path
from typing import List, Literal, Tuple

import pandas as pd
import torch
from lightning_fabric.utilities import move_data_to_device
from matplotlib import pyplot as plt
from torchvision.io import read_image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.ops import box_area, box_iou
from torchvision.ops.boxes import _box_inter_union
from tqdm import tqdm
from transformers import TableTransformerForObjectDetection

from src.historicdocumentprocessing.util.tablesutil import reversetablerelativebboxes_outer, boxoverlap
from src.historicdocumentprocessing.tabletransformer_dataset import CustomDataset
from src.historicdocumentprocessing.util.plottotikz import save_plot_as_tikz


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
        predious, _, tp, fp, _ = calcstats_iou(
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
        predious, _, tp, fp, _ = calcstats_iou(
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


def calcstats_iodt(
    predbox: torch.tensor,
    targetbox: torch.tensor,
    imname: str = None,
    iou_thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate tp, tp, fn based on IoDT (Intersection over Detection) at given IoU Thresholds
    Args:
        predbox:
        targetbox:
        imname:
        iou_thresholds:

    Returns:

    """
    threshold_tensor = torch.tensor(iou_thresholds)
    if not predbox.numel():
        # print(predbox.shape)
        warnings.warn(
            f"A Prediction Bounding Box Tensor in {imname} is empty. (no predicitons)"
        )
        return (
            torch.zeros(predbox.shape[0]),
            torch.zeros(targetbox.shape[0]),
            torch.zeros(threshold_tensor.shape),
            torch.zeros(threshold_tensor.shape),
            torch.full(threshold_tensor.shape, fill_value=targetbox.shape[0]),
        )
    iodt = torch.nan_to_num(intersection_over_detection(predbox, targetbox))
    # print(IoDT, IoDT.shape)
    prediodt = iodt.amax(dim=1)
    targetiodt = iodt.amax(dim=0)
    # print(predbox.shape, prediodt.shape, prediodt)
    # print(targetbox.shape, targetiodt.shape, targetiodt)
    # print(imname)
    assert len(prediodt) == predbox.shape[0]
    assert len(targetiodt) == targetbox.shape[0]
    tp = torch.sum(
        prediodt.unsqueeze(-1).expand(-1, len(threshold_tensor)) >= threshold_tensor,
        dim=0,
    )
    # print(tp, prediodt.unsqueeze(-1).expand(-1, len(threshold_tensor)))
    fp = iodt.shape[0] - tp
    fn = torch.sum(
        targetiodt.unsqueeze(-1).expand(-1, len(threshold_tensor)) < threshold_tensor,
        dim=0,
    )
    assert torch.equal(
        fp,
        torch.sum(
            prediodt.unsqueeze(-1).expand(-1, len(threshold_tensor)) < threshold_tensor,
            dim=0,
        ),
    )
    # print(tp, fp, fn)
    return prediodt, targetiodt, tp, fp, fn


def intersection_over_detection(predbox, targetbox):
    """
    Calculate the IoDT (Intersection over Detection Metric): Intersection of prediction and target over the total prediction area
    Args:
        predbox:
        targetbox:

    Returns:

    """
    inter, union = _box_inter_union(targetbox, predbox)
    predarea = box_area(predbox)
    # print(inter, inter.shape)
    # print(predarea, predarea.shape)
    iodt = torch.div(inter, predarea)
    # print(IoDT, IoDT.shape)
    iodt = iodt.T
    return iodt


def calcstats_overlap(
    predbox: torch.tensor, targetbox: torch.tensor, imname: str = None, fuzzy: int = 25
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates stats based on wether a prediciton box fully overlaps with a target box instead of IoU
    tp = prediciton lies fully within a target box
    fp = prediciton does not lie fully within a target box
    fn = target boxes that do not have a prediciton lying fully within them
    Args:
        fuzzy:
        predbox:
        targetbox:
        imname:

    Returns:


    """
    # print(predbox.shape, targetbox.shape)
    if not predbox.numel():
        # print(predbox.shape)
        warnings.warn(
            f"A Prediction Bounding Box Tensor in {imname} is empty. (no predicitons)"
        )
        return (
            torch.zeros(1),
            torch.zeros(1),
            torch.full([1], fill_value=targetbox.shape[0]),
        )
    overlapmat = torch.zeros((predbox.shape[0], targetbox.shape[0]))
    for i, pred in enumerate(predbox):
        for j, target in enumerate(targetbox):
            if boxoverlap(bbox=pred, tablebox=target, fuzzy=fuzzy):
                # print(imname, pred,target)
                overlapmat[i, j] = 1
    predious = overlapmat.amax(dim=1)
    targetious = overlapmat.amax(dim=0)
    tp = torch.sum(predious.unsqueeze(-1), dim=0)
    # print(tp, predious.unsqueeze(-1).expand(-1, len(threshold_tensor)))
    fp = overlapmat.shape[0] - tp
    # print(targetious)
    # print(torch.where((targetious.unsqueeze(-1)==0), 1.0, 0.0).flatten())
    fn = torch.sum(targetious.unsqueeze(-1) < 1, dim=0)
    # print(overlapmat.shape, predbox.shape, targetbox.shape)
    # print(tp, fp, fn)
    # print(type(tp))
    # print(tp.shape)
    return tp, fp, fn


def calcmetric_overlap(
    tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate Metrics from true positives, false positives, false negatives based on wether a prediciton box fully
    overlaps with a target box

    Args:
        tp: true positives
        fp: false positives
        fn: false negatives

    Returns:


    """
    precision = torch.nan_to_num(tp / (tp + fp))
    recall = torch.nan_to_num(tp / (tp + fn))
    f1 = torch.nan_to_num(2 * (precision * recall) / (precision + recall))
    return precision, recall, f1


def calcstats_iou(
    predbox: torch.tensor,
    targetbox: torch.tensor,
    iou_thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9],
    imname: str = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculates IoU and resulting tp, fp, fn.

    Calculates the intersection over union as well as resulting tp, fp, fn at given IoU Thresholds for the bounding boxes
    of a target and prediciton given as torch tensor with bounding box
    coordinates in format x_min, y_min, x_max, y_max

    Args:
        predbox: predictions
        targetbox: targets
        iou_thresholds: List of IoU Thresholds

    Returns:
        tuple of predious, targetious, tp, fp, fn

    """
    threshold_tensor = torch.tensor(iou_thresholds)
    if not predbox.numel():
        warnings.warn(
            f"A Prediction Bounding Box Tensor in {imname} is empty. (no predicitons)"
        )
        return (
            torch.zeros(predbox.shape[0]),
            torch.zeros(targetbox.shape[0]),
            torch.zeros(threshold_tensor.shape),
            torch.zeros(threshold_tensor.shape),
            torch.full(threshold_tensor.shape, fill_value=targetbox.shape[0]),
        )
    ioumat = torch.nan_to_num(box_iou(predbox, targetbox))
    predious = ioumat.amax(dim=1)
    targetious = ioumat.amax(dim=0)
    assert len(predious) == predbox.shape[0]
    assert len(targetious) == targetbox.shape[0]

    tp = torch.sum(
        predious.unsqueeze(-1).expand(-1, len(threshold_tensor)) >= threshold_tensor,
        dim=0,
    )

    fp = ioumat.shape[0] - tp
    fn = torch.sum(
        targetious.unsqueeze(-1).expand(-1, len(threshold_tensor)) < threshold_tensor,
        dim=0,
    )
    assert torch.equal(
        fp,
        torch.sum(
            predious.unsqueeze(-1).expand(-1, len(threshold_tensor)) < threshold_tensor,
            dim=0,
        ),
    )
    return predious, targetious, tp, fp, fn


def calcmetric(
    tp: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
    iou_thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Calculate Metrics from true positives, false positives, false negatives at given iou thresholds.

    Args:
        tp: True Positives
        fp: False Positives
        fn: False Negatives
        iou_thresholds: List of IoU Thresholds

    Returns:
        precision, recall, f1, wf1

    """
    threshold_tensor = torch.tensor(iou_thresholds)
    precision = torch.nan_to_num(tp / (tp + fp))
    recall = torch.nan_to_num(tp / (tp + fn))
    f1 = torch.nan_to_num(2 * (precision * recall) / (precision + recall))
    wf1 = f1 @ threshold_tensor / torch.sum(threshold_tensor)
    return precision, recall, f1, wf1


def get_dataframe(
    fnsum,
    fpsum,
    tpsum,
    nopredcount: int = None,
    imnum: int = None,
    iou_thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9],
):
    totalfullprec, totalfullrec, totalfullf1, totalfullwf1 = calcmetric(
        tpsum, fpsum, fnsum, iou_thresholds=iou_thresholds
    )
    totalfullmetrics = {"wf1": totalfullwf1.item()}
    totalfullmetrics.update({f"Number of evaluated files": imnum})
    totalfullmetrics.update({f"Evaluated files without predictions:": nopredcount})
    totalfullmetrics.update(
        {
            f"f1@{iou_thresholds[i]}": totalfullf1[i].item()
            for i in range(len(iou_thresholds))
        }
    )
    totalfullmetrics.update(
        {
            f"prec@{iou_thresholds[i]}": totalfullprec[i].item()
            for i in range(len(iou_thresholds))
        }
    )
    totalfullmetrics.update(
        {
            f"recall@{iou_thresholds[i]}": totalfullrec[i].item()
            for i in range(len(iou_thresholds))
        }
    )
    totalfullmetrics.update(
        {f"tp@{iou_thresholds[i]}": tpsum[i].item() for i in range(len(iou_thresholds))}
    )
    totalfullmetrics.update(
        {f"fp@{iou_thresholds[i]}": fpsum[i].item() for i in range(len(iou_thresholds))}
    )
    totalfullmetrics.update(
        {f"fn@{iou_thresholds[i]}": fnsum[i].item() for i in range(len(iou_thresholds))}
    )
    return totalfullmetrics
