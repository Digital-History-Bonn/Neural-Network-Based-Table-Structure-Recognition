import glob
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from torchvision.ops.boxes import _box_inter_union

def getcells(
    rows: torch.Tensor, cols: torch.Tensor, keepnonoverlap: bool = True
) -> torch.Tensor:
    """get cell bboxes by intersecting row and column bboxes (for tabletransformer)"""
    # print("rows:",rows.shape)
    # print("cols:",cols.shape)
    inter, _ = _box_inter_union(rows, cols)
    newcells = []
    for rowidx, colidx in inter.nonzero():
        minxmaxymax = torch.min(rows[rowidx, 2:], cols[colidx, 2:])
        maxxminymin = torch.max(rows[rowidx, :2], cols[colidx, :2])
        newcell = torch.hstack([maxxminymin, minxmaxymax])
        # print(newcell.shape)
        newcells.append(newcell)
    if keepnonoverlap:
        if cols.shape[0] > 0:
            for rowidx in torch.where(inter.amax(dim=1) <= 0, 1, 0).nonzero():
                newcells.append(rows[rowidx])
        else:
            for row in rows:
                newcells.append(row)
        if rows.shape[0] > 0:
            for colidx in torch.where(inter.amax(dim=0) <= 0, 1, 0).nonzero():
                newcells.append(cols[colidx])
        else:
            for col in cols:
                newcells.append(col)
    return torch.vstack(newcells) if newcells else torch.empty(0, 4)


def avrgeuc(boxes: torch.Tensor) -> float:
    count = 0
    dist = 0
    for box1 in boxes:
        singledist = 0
        for box2 in boxes:
            if not torch.equal(box1, box2):
                # print("j")
                new = eucsimilarity(box1.numpy(), box2.numpy())
                if not singledist or 0 < new < singledist:
                    singledist = new
            # singledist = abs(math.sqrt(pow((abs(box1[0]-box2[2])),2)+pow(abs(box1[3]-box2[1]),2)))
        #        print("f")
        dist += singledist
        count += 1
    # if dist==0:
    #    print(boxes, dist)
    # print(dist, count)
    if count != 0 and dist == 0:
        # print(boxes, dist)
        return 1
    elif count == 0:
        return 0
    return dist / count


def eucsimilarity(x, y):
    # print(x[0], y, (x[2]-x[0]))
    # print(pow(torch.norm(torch.max(torch.zeros(2), x[:2]-y[2:])),2))
    # print(pow(torch.max(torch.zeros(2), y[:2]-x[2:]),2))
    # print(np.where((x[:2]-y[2:])<0,0, (x[:2]-y[2:])), x[:2]-y[2:])
    slice = int(x.shape[0] / 2)
    # print(slice)
    res = math.sqrt(
        pow(
            np.linalg.norm(
                np.where((x[:slice] - y[slice:]) < 0, 0, (x[:slice] - y[slice:]))
            ),
            2,
        )
        + pow(
            np.linalg.norm(
                np.where((y[:slice] - x[slice:]) < 0, 0, y[:slice] - x[slice:])
            ),
            2,
        )
    )
    # res = math.sqrt(pow(np.linalg.norm(np.where((x[:2]-y[2:])<0,0, (x[:2]-y[2:]))),2)+pow(np.linalg.norm(np.where((y[:2]-x[2:])<0,0, y[:2]-x[2:])),2))
    # res = abs(math.sqrt(pow(x[2]-x[0],2)+pow((x[3]-x[1]),2))-math.sqrt(pow(y[2]-y[0],2)+pow(y[3]-y[1],2)))
    # print(x, y)
    # print(res)
    return res


def clustertables(boxes: torch.Tensor, epsprefactor: float = 1 / 6):
    tables = []
    eps = avrgeuc(boxes)
    # print("eps: ", eps)
    if eps:
        clustering = DBSCAN(
            eps=(epsprefactor) * eps, min_samples=2, metric=eucsimilarity
        ).fit(boxes.numpy())
        for label in set(clustering.labels_):
            table = boxes[clustering.labels_ == label]
            # print(label, clustering.labels_)
            tables.append(table)
    return tables


def clustertablesseperately(
    boxes: torch.Tensor,
    epsprefactor: Tuple[float, float] = tuple([3, 1.5]),
    includeoutlier: bool = True,
    minsamples: List[int] = [4, 5],
):
    tables = []
    xtables = []
    # print(boxes.shape)
    xboxes = boxes[:, [0, 2]]
    # yboxes = boxes[:,[1,3]]
    # print(boxes[:,[0,2]].shape)
    xdist = avrgeuc(xboxes)
    # ydist = avrgeuc(yboxes)
    #    print("h")
    epsprefactor1 = epsprefactor[0]
    epsprefactor2 = epsprefactor[1]
    if xdist:
        clustering = DBSCAN(
            eps=(epsprefactor1) * xdist, min_samples=minsamples[0], metric=eucsimilarity
        ).fit(xboxes.numpy())
        for label in set(clustering.labels_):
            xtable = boxes[clustering.labels_ == label]
            # print(label, clustering.labels_)
            if includeoutlier or (int(label) != -1):
                xtables.append(xtable)
            else:
                print(label)
            # print(xtable)
        for prototable in xtables:
            yboxes = prototable[:, [1, 3]]
            ydist = avrgeuc(yboxes)
            if ydist:
                clustering = DBSCAN(
                    eps=(epsprefactor2) * ydist,
                    min_samples=minsamples[1],
                    metric=eucsimilarity,
                ).fit(yboxes.numpy())
                for label in set(clustering.labels_):
                    table = prototable[(clustering.labels_ == label)]
                    # print(label, clustering.labels_)
                    if includeoutlier or (int(label) != -1):
                        tables.append(table)
                    else:
                        print(label)
            #                    print("y")
            elif len(prototable) == 1:
                tables.append(prototable)
    return tables


def getsurroundingtable(boxes: torch.Tensor) -> torch.Tensor:
    """get surrounding table of a group of bounding boxes given as torch.Tensor
    Args:
        boxes: (cell) bounding boxes as torch.Tensor

    Returns: surrounding table"""
    return torch.hstack(
        [
            torch.min(boxes[:, 0]),
            torch.min(boxes[:, 1]),
            torch.max(boxes[:, 2]),
            torch.max(boxes[:, 3]),
        ]
    )


def gettablerelativebbox(box: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    # print(box, table)
    return torch.hstack(
        [box[0] - table[0], box[1] - table[1], box[2] - table[0], box[3] - table[1]]
    )


def gettablerelativebboxes(boxes: torch.Tensor) -> torch.Tensor:
    tablecoords = getsurroundingtable(boxes)
    # print(tablecoords)
    newboxes = []
    for box in boxes:
        newboxes.append(gettablerelativebbox(box, tablecoords))
    # print(newboxes)
    return torch.vstack(newboxes)


if __name__ == "__main__":
    """
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/tabletransformer/testevalfinal1/fullimg/BonnData/tabletransformer_v0_new_BonnDataFullImage_tabletransformer_estest_BonnData_fullimage_e250_valid_es.pt/tableareaonly/no_filtering_iou_0.5_0.9/cells/fullimageiodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/tabletransformer/testevalfinal1/fullimg/BonnData/tabletransformer_v0_new_BonnDataFullImage_tabletransformer_estest_BonnData_fullimage_e250_valid_es.pt/tableareaonly/no_filtering_iou_0.5_0.9/cells/fullimageiou.csv",
        resultmetric="iou")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/tabletransformer/testevalfinal1/fullimg/BonnData/tabletransformer_v0_new_BonnDataFullImage_tabletransformer_estest_BonnData_fullimage_e250_valid_es.pt/tableareaonly/no_filtering_iou_0.5_0.9/fullimageiodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/tabletransformer/testevalfinal1/fullimg/BonnData/tabletransformer_v0_new_BonnDataFullImage_tabletransformer_estest_BonnData_fullimage_e250_valid_es.pt/tableareaonly/no_filtering_iou_0.5_0.9/fullimageiou.csv",
        resultmetric="iou")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/tabletransformer/testevalfinal1/fullimg/BonnData/tabletransformer_v0_new_BonnDataFullImage_tabletransformer_estest_BonnData_fullimage_e250_valid_end.pt/tableareaonly/no_filtering_iou_0.5_0.9/cells/fullimageiodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/tabletransformer/testevalfinal1/fullimg/BonnData/tabletransformer_v0_new_BonnDataFullImage_tabletransformer_estest_BonnData_fullimage_e250_valid_end.pt/tableareaonly/no_filtering_iou_0.5_0.9/cells/fullimageiou.csv",
        resultmetric="iou")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/tabletransformer/testevalfinal1/fullimg/BonnData/tabletransformer_v0_new_BonnDataFullImage_tabletransformer_estest_BonnData_fullimage_e250_valid_end.pt/tableareaonly/no_filtering_iou_0.5_0.9/fullimageiodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/tabletransformer/testevalfinal1/fullimg/BonnData/tabletransformer_v0_new_BonnDataFullImage_tabletransformer_estest_BonnData_fullimage_e250_valid_end.pt/tableareaonly/no_filtering_iou_0.5_0.9/fullimageiou.csv",
        resultmetric="iou")
    """
    """
    BonnTablebyCat(resultfile=f"{Path(__file__).parent.absolute()}/../../../results/kosmos25/testevalfinal1/BonnData_Tables/iou_0.5_0.9/tableareaonly/fullimageiodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/kosmos25/testevalfinal1/BonnData_Tables/iou_0.5_0.9/tableareaonly/fullimageiou.csv",
        resultmetric="iou")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/fullimg/BonnData/BonnDataFullImage1_BonnData_fullimage_e250_es.pt/tableareaonly/iou_0.5_0.9/fullimageiodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/fullimg/BonnData/BonnDataFullImage1_BonnData_fullimage_e250_es.pt/iou_0.5_0.9/fullimageiodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/fullimg/BonnData/BonnDataFullImage1_BonnData_fullimage_e250_es.pt/tableareaonly/iou_0.5_0.9/fullimageiou.csv",
        resultmetric="iou")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/fullimg/BonnData/BonnDataFullImage1_BonnData_fullimage_e250_es.pt/iou_0.5_0.9/fullimageiou.csv",
        resultmetric="iou")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_es.pt/tableareaonly/iou_0.5_0.9/fullimageiodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_es.pt/iou_0.5_0.9/fullimageiodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_es.pt/tableareaonly/iou_0.5_0.9/fullimageiou.csv",
        resultmetric="iou")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_es.pt/iou_0.5_0.9/fullimageiou.csv",
        resultmetric="iou")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_end.pt/tableareaonly/iou_0.5_0.9/fullimageiodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_end.pt/iou_0.5_0.9/fullimageiodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_end.pt/tableareaonly/iou_0.5_0.9/fullimageiou.csv",
        resultmetric="iou")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_end.pt/iou_0.5_0.9/fullimageiou.csv",
        resultmetric="iou")


    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/tableareacutout/BonnData/run3_BonnData_cell_aug_loadrun_GloSAT_cell_aug_e250_es_e250_es.pt/iodt.csv")

    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/tableareacutout/BonnData/run3_BonnData_cell_loadrun_GloSAT_cell_e250_es_e250_es.pt/iodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/tableareacutout/BonnData/run_BonnData_cell_e250_es.pt/iodt.csv")

    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/tableareacutout/BonnData/run3_BonnData_cell_aug_loadrun_GloSAT_cell_aug_e250_es_e250_es.pt/iou.csv",
        resultmetric="iou")

    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/tableareacutout/BonnData/run3_BonnData_cell_loadrun_GloSAT_cell_e250_es_e250_es.pt/iou.csv",
        resultmetric="iou")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testevalfinal1/tableareacutout/BonnData/run_BonnData_cell_e250_es.pt/iou.csv",
        resultmetric="iou")


    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, **{"box_detections_per_img": 200}
    )
    model.load_state_dict(
        torch.load(
            f"{Path(__file__).parent.absolute()}/../../../checkpoints/fasterrcnn/BonnDataFullImage1_BonnData_fullimage_e250_es.pt"
        )
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        model.eval()
    else:
        print("Cuda not available")
        exit()
    #boxes = torch.load(f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/BonnData/I_HA_Rep_89_Nr_16160_0170/I_HA_Rep_89_Nr_16160_0170.pt")
    img = (read_image(f"{Path(__file__).parent.absolute()}/../../../data/BonnData/test/I_HA_Rep_89_Nr_16160_0170/I_HA_Rep_89_Nr_16160_0170.jpg") / 255).to(device)
    output = model([img])
    output = {k: v.detach().cpu() for k, v in output[0].items()}
    boxes= output['boxes']
    tables = clustertables(boxes, epsprefactor=1/6)
    img = read_image(
        f"{Path(__file__).parent.absolute()}/../../../data/BonnData/test/I_HA_Rep_89_Nr_16160_0170/I_HA_Rep_89_Nr_16160_0170.jpg")
    for i,t in enumerate(tables):
       res = draw_bounding_boxes(image=img, boxes=t)
       res = Image.fromarray(res.permute(1, 2, 0).numpy())
       # print(f"{savepath}/{identifier}.jpg")
       res.save(f"{Path(__file__).parent.absolute()}/../../../images/test/test_rcnn_{i}.jpg")

    """
    """
    with open(
        f"{Path(__file__).parent.absolute()}/../../../results/kosmos25/BonnData/Tabellen/test/I_HA_Rep_89_Nr_16160_0170/I_HA_Rep_89_Nr_16160_0170.jpg.json"
    ) as p:
        boxes = extractboxes(json.load(p))
    # tables = clustertables(boxes)
    tables = clustertablesseperately(boxes, includeoutlier=False)
    img = read_image(
        f"{Path(__file__).parent.absolute()}/../../../data/BonnData/test/I_HA_Rep_89_Nr_16160_0170/I_HA_Rep_89_Nr_16160_0170.jpg"
    )
    for i, t in enumerate(tables):
        res = draw_bounding_boxes(image=img, boxes=t)
        res = Image.fromarray(res.permute(1, 2, 0).numpy())
        # print(f"{savepath}/{identifier}.jpg")
        res.save(
            f"{Path(__file__).parent.absolute()}/../../../images/test/test_{i}.jpg"
        )
    """



def remove_invalid_bbox(box, impath: str = "") -> torch.Tensor:
    """Function to remove BoundingBoxes with invalid coordinates.

       Args:
           box: BoundingBoxes
           impath: path to image that BoundingBoxes were detected on (to localize where invalid BBox was detected)

       Returns:
           tensor of valid BoundingBoxes
       """
    newbox = []
    for b in box:
        if b[0] < b[2] and b[1] < b[3]:
            newbox.append(b)
        else:
            print(f"Invalid bounding box in image {impath.split('/')[-1]}", b)
    return torch.vstack(newbox) if newbox else torch.empty(0, 4)


def reversetablerelativebboxes_inner(
    tablebbox: torch.Tensor, cellbboxes: torch.Tensor
) -> torch.Tensor:
    """Returns bounding boxes relative to image rather than relative to table so that evaluation of bounding box accuracy is possible on whole image.

    inner function for one bounding box

    Args:
        tablebbox: table coordinates
        cellbboxes: cell coordinates relative to table

    Returns:

    """
    return cellbboxes + torch.tensor(
        data=[tablebbox[0], tablebbox[1], tablebbox[0], tablebbox[1]]
    )


def reversetablerelativebboxes_outer(fpath: str) -> torch.Tensor:
    """Returns bounding boxes relative to image rather than relative to table so that evaluation of bounding box accuracy is possible on whole image.

    Args:
        fpath: path to folder with preprocessed BBox files

    Returns:
        tensor with all BBoxes in image relative to image
    """
    tablebboxes = torch.load(glob.glob(f"{fpath}/*tables.pt")[0])
    newcoords = torch.zeros((0, 4))
    for table in glob.glob(f"{fpath}/*_cell_*.pt"):
        n = int(table.split(".")[-2].split("_")[-1])
        newcell = reversetablerelativebboxes_inner(tablebboxes[n], torch.load(table))
        newcoords = torch.vstack((newcoords, newcell))
    return newcoords


def boxoverlap(
    bbox: Tuple[int, int, int, int],
    tablebox: Tuple[int, int, int, int],
    fuzzy: int = 25,
) -> bool:
    """
    Checks if bbox lies in tablebox
    Args:
        bbox: the Bounding Box
        tablebox: the Bounding Box that bbox should lie in
        fuzzy: fuzzyness of tablebox boundaries
    Returns:

    """
    return (
        bbox[0] >= (tablebox[0] - fuzzy)
        and bbox[1] >= (tablebox[1] - fuzzy)
        and bbox[2] <= (tablebox[2] + fuzzy)
        and bbox[3] <= (tablebox[3] + fuzzy)
    )


def extractboundingbox(bbox: dict) -> Tuple[int, int, int, int]:
    """Get singular bounding box from dict.

    Args:
        bbox: dict of bounding box coordinates

    Returns: Tuple of bounding box coordinates in format x_min, y_min, x_max, y_max

    """
    return int(bbox["x0"]), int(bbox["y0"]), int(bbox["x1"]), int(bbox["y1"])


def extractboxes(boxdict: dict, fpath: str = None) -> torch.Tensor:
    """Takes a Kosmos2.5-Output-Style Dict and extracts the bounding boxes.

    Args:
        fpath: path to table folder (if only bboxes in a table wanted)
        boxdict(dict): dictionary in style of kosmos output (from json)

    Returns:
        bounding boxes as torch tensor

    """
    boxlist = []
    tablebboxes = None
    if fpath:
        tablebboxes = torch.load(glob.glob(f"{fpath}/*tables.pt")[0])
        # print(tablebboxes)
        # print(tablebboxes)
    for box in boxdict["results"]:
        bbox = extractboundingbox(box["bounding box"])
        # print(bbox)
        if fpath:
            for table in tablebboxes:
                if boxoverlap(bbox, table):
                    boxlist.append(bbox)
        else:
            boxlist.append(bbox)
    # print(boxlist)
    return torch.tensor(boxlist) if boxlist else torch.empty(0, 4)
