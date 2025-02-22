from pathlib import Path

import pandas
import torch
from bs4 import BeautifulSoup
from torchvision.ops import box_iou

from src.historicdocumentprocessing.dataprocessing import processdata_wildtable_inner
from src.historicdocumentprocessing.kosmos_eval import (
    calcmetric,
    calcstats_IoU,
    get_dataframe,
    reversetablerelativebboxes_outer,
)


def test_ioumetrics():

    testpred = torch.tensor([[0, 0, 2, 2], [1, 0, 2, 2], [5, 5, 5.5, 6.5]])
    testground = torch.tensor([[0, 0, 2, 2], [0, 0, 1, 1], [0, 0, 2, 1], [7, 7, 9, 8]])
    # print("predboxes:", testpred)
    # print("targetboxes:", testground)
    ioumat = box_iou(testpred, testground)
    ioumattrue = torch.tensor([[1.0, 0.25, 0.5, 0], [0.5, 0, 1 / 3, 0], [0, 0, 0, 0]])
    assert torch.equal(ioumat, ioumattrue)
    # print("ioumat:",ioumat)

    # test with pg calc
    iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_tensor = torch.tensor(iou_thresholds)
    prediousp = ioumat.amax(dim=1)
    targetiousp = ioumat.amax(dim=0)
    n_pred, n_target = ioumat.shape
    tpg = torch.sum(
        prediousp.expand(len(iou_thresholds), n_pred) >= threshold_tensor[:, None],
        dim=1,
    )
    fpg = len(ioumat) - tpg

    assert torch.equal(
        fpg,
        torch.sum(
            prediousp.expand(len(iou_thresholds), n_pred) < threshold_tensor[:, None],
            dim=1,
        ),
    )

    fng = torch.sum(
        targetiousp.expand(len(iou_thresholds), n_target) < threshold_tensor[:, None],
        dim=1,
    )
    #

    precisiong, recallg, f1g, wf1g = calcmetric(tpg, fpg, fng)
    # print("predious with pg calc:", prediousp)
    # print("targetious with pg calc:", targetiousp)
    # print("tp,fp,fn with pg calc:", tpg, fpg, fng)
    # print("precision, recall with pg calc:", precisiong, recallg)
    # print("f1, wf1: with pg calc", f1g, wf1g)
    predious, targetious, tp, fp, fn = calcstats_IoU(testpred, testground)
    precision, recall, f1, wf1 = calcmetric(tp, fp, fn)
    # print("predious:", predious)
    # print("targetious:", targetious)
    # print("tp,fp,fn:", tp,fp,fn)
    # print("precision, recall:", precision, recall)
    # print("f1, wf1:", f1, wf1)
    tpc = torch.Tensor([2, 1, 1, 1, 1])
    fpc = torch.Tensor([1, 2, 2, 2, 2])
    fnc = torch.Tensor([2, 3, 3, 3, 3])
    precisionc = torch.Tensor([2 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3])
    recallc = torch.Tensor([0.5, 1 / 4, 1 / 4, 1 / 4, 1 / 4])
    f1c = torch.Tensor([4 / 7, 2 / 7, 2 / 7, 2 / 7, 2 / 7])
    assert torch.equal(tpc, tpg)
    assert torch.equal(tpc, tp)
    assert torch.equal(fpc, fpg)
    assert torch.equal(fpc, fp)
    assert torch.equal(fnc, fng)
    assert torch.equal(fnc, fn)
    assert torch.equal(precisionc, precision)
    assert torch.equal(recallc, recall)
    assert torch.allclose(f1c, f1)
    assert torch.equal(precision, precisiong)
    assert torch.equal(recall, recallg)
    assert torch.equal(f1, f1g)
    assert torch.equal(wf1, wf1g)
    df = get_dataframe(fn, fp, tp)
    wf1c = (f1c @ threshold_tensor) / threshold_tensor.sum()
    assert df["wf1"] == wf1, "result of calc steps must be same as result in dataframe"
    assert df["wf1"] == wf1g, "result of calc steps must be same as result in dataframe"
    assert (df["wf1"] - wf1c.item()) < 1e-9, "result must be same as manual calculation"
    # print("dataframe function test: ", df)


def test_iousum():
    """
    wf1 über summe aller tp,fp,fn berechnen nicht als mittel der wf1 werte????
    Returns:

    """
    df = pandas.read_csv(
        f"{Path(__file__).parent.absolute()}/../results/fasterrcnn/testevalfinal1/fullimg/Tablesinthewild/testseveralcalls_5_without_valid_split_Tablesinthewild_fullimage_e50_es.pt/iou_0.5_0.9/fullimageiou.csv"
    )
    df1 = get_dataframe(
        fnsum=torch.Tensor([df["fn@0.5"].sum()]),
        fpsum=torch.Tensor([df["fp@0.5"].sum()]),
        tpsum=torch.Tensor([df["tp@0.5"].sum()]),
        iou_thresholds=[0.5],
    )
    df2 = pandas.read_csv(
        f"{Path(__file__).parent.absolute()}/../results/fasterrcnn/testevalfinal1/fullimg/Tablesinthewild/testseveralcalls_5_without_valid_split_Tablesinthewild_fullimage_e50_es.pt/iou_0.5_0.9/overview.csv"
    )
    assert df2.loc[1, "f1@0.5"] == df1["f1@0.5"], "calculated total  f"


def test_inputdata():
    full = torch.load(
        f"{Path(__file__).parent.absolute()}/../data/Tablesinthewild/preprocessed/curved/table_spider_00909/table_spider_00909.pt"
    )
    fromtablerelative = reversetablerelativebboxes_outer(
        f"{Path(__file__).parent.absolute()}/../data/Tablesinthewild/preprocessed/curved/table_spider_00909"
    )
    ground = processdata_wildtable_inner(
        f"{Path(__file__).parent.absolute()}/../data/Tablesinthewild/rawdata/test/test-xml-revise/test-xml-revise/table_spider_00909.xml"
    )
    assert torch.equal(
        full, fromtablerelative
    ), "table relative and non table relative bounding boxes should be the same when calculating back to whole image"
    assert torch.equal(
        ground, full
    ), "processed ground truth bounding boxes should be identical to unprocessed ground truth bounding boxes"
    i = 0
    bboxes = []
    with open(
        f"{Path(__file__).parent.absolute()}/../data/Tablesinthewild/rawdata/test/test-xml-revise/test-xml-revise/table_spider_00909.xml"
    ) as d:
        xml = BeautifulSoup(d, "xml")
        for box in xml.find_all("bndbox"):
            new = torch.tensor(
                [
                    int(float(box.xmin.get_text())),
                    int(float(box.ymin.get_text())),
                    int(float(box.xmax.get_text())),
                    int(float(box.ymax.get_text())),
                ],
                dtype=torch.float,
            )
            if new[0] < new[2] and new[1] < new[3]:
                i += 1
                bboxes.append(new)
    assert torch.equal(
        ground, torch.vstack(bboxes)
    ), "unprocessed and processed ground truth should be identical"
    fulltest = torch.load(
        f"{Path(__file__).parent.absolute()}/../data/Tablesinthewild/test/table_spider_00909/table_spider_00909.pt"
    )
    assert torch.equal(
        full, fulltest
    ), "the ground truth bounding boxes used for full testing and testing by category should be identical"


if __name__ == "__main__":
    # test_inputdata()
    # test_iousum()
    test_ioumetrics()
