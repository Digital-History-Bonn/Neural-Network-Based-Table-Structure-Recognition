from pathlib import Path

import torch
from PIL import Image
from lightning_fabric.utilities import move_data_to_device
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import draw_bounding_boxes
from transformers import AutoModelForObjectDetection

from src.historicdocumentprocessing.tabletransformer_dataset import CustomDataset
from src.historicdocumentprocessing.tabletransformer_train import TableTransformer


def inferencetest(imgpath:str,modelpath:str="microsoft/table-transformer-structure-recognition-v1.1-all",  num:int=0):
    device = (
        torch.device(f"cuda:0")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model = AutoModelForObjectDetection.from_pretrained(modelpath).to(device)
    dataset = CustomDataset(f"{Path(__file__).parent.absolute()}/../data/BonnData/test")
    example_image = Image.open(imgpath).convert("RGB")
    print(model.config.id2label)
    model.eval()
    encoding = move_data_to_device(dataset.ImageProcessor(example_image, return_tensors="pt"), device)

    with torch.no_grad():
        out = model(**encoding)
    width, height = example_image.size
    pred = dataset.ImageProcessor.post_process_object_detection(out, target_sizes=[(height, width)])[0]
    model.train()
    boxes = {
        "prediction": pred["boxes"].detach().cpu(),
    }
    colors = ["green" for i in range(boxes["prediction"].shape[0])]
    labels = pred["labels"].detach().cpu()
    print(labels)
    result = draw_bounding_boxes(
        image=pil_to_tensor(example_image).to(torch.uint8),
        boxes=boxes["prediction"],
        colors=colors,
        labels=[str(model.config.id2label[label]) for label in labels.tolist()])
    img = Image.fromarray(result.permute(1, 2, 0).numpy())
    img.save(f"{Path(__file__).parent.absolute()}/../images/test/testtabletransformerinference_{num}.jpg")

def inferencetest2(imgpath:str,modelpath:str=None,  num:int=0):
    device = (
        torch.device(f"cuda:0")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("h")
    traindataset = CustomDataset(
        f"{Path(__file__).parent.absolute()}/../data/BonnData/train"
    )

    validdataset = CustomDataset(
            f"{Path(__file__).parent.absolute()}/../data/BonnData/valid"
        )
    #    valid_dataloader = DataLoader(dataset=validdataset)
    # train_dataloader = DataLoader(dataset=traindataset)
    testdataset = CustomDataset(f"{Path(__file__).parent.absolute()}/../data/BonnData/test")
    model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition-v1.1-all").to(device)
    if modelpath:
        model.load_state_dict(torch.load(modelpath))
    #model = TableTransformer(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4,
    #                         traindataset=traindataset, valdataset=validdataset, testdataset=testdataset,
    #                         datasetname="BonnData")
    #print("f")
    #model.load_state_dict(torch.load(modelpath))
    #model = TableTransformer.load_from_checkpoint(modelpath, traindataset=traindataset, testdataset=testdataset, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, valdataset=validdataset,datasetname="BonnData", strict=False)
    dataset = CustomDataset(f"{Path(__file__).parent.absolute()}/../data/BonnData/test")
    #example_image = Image.open(imgpath).convert("RGB")
    image = Image.open(imgpath).convert("RGB")
    example_image, _, _ = testdataset.getimgtarget(0)
    dataloader = DataLoader(testdataset, batch_size=1, collate_fn=testdataset.collate_fn)
    batch = next(iter(dataloader))
    print(model.config.id2label)
    model.eval()
    encoding = move_data_to_device(dataset.ImageProcessor(example_image, return_tensors="pt"), device)
    pixelvalues = dataset.ImageProcessor(example_image, return_tensors="pt", do_pad=False)["pixel_values"]
    encoding1 = dataset.ImageProcessor.pad(pixelvalues, return_tensors="pt")
    encoding2 = move_data_to_device(dataset.ImageProcessor(image, return_tensors="pt"), device)
    with torch.no_grad():
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
        print(labels)
        #print(encoding)
        print(pixel_values)
        print(encoding1["pixel_values"], encoding["pixel_values"])
        out = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        out1 = model(**encoding2)
        #out = model(**encoding)
    width, height = example_image.size
    pred = dataset.ImageProcessor.post_process_object_detection(out, target_sizes=[(height, width)])[0]
    boxes = {
        "prediction": pred["boxes"].detach().cpu(),
    }
    colors = ["green" for i in range(boxes["prediction"].shape[0])]
    labels = pred["labels"].detach().cpu()
    print(labels)
    result = draw_bounding_boxes(
        image=pil_to_tensor(example_image).to(torch.uint8),
        boxes=boxes["prediction"],
        colors=colors,
        labels=[str(model.config.id2label[label]) for label in labels.tolist()])
    img = Image.fromarray(result.permute(1, 2, 0).numpy())
    img.save(f"{Path(__file__).parent.absolute()}/../images/test/testtabletransformerinference_{num}.jpg")
    ####
    width, height = image.size
    pred1 = dataset.ImageProcessor.post_process_object_detection(out1, target_sizes=[(height, width)])[0]
    boxes1 = {
        "prediction": pred1["boxes"].detach().cpu(),
    }
    colors1 = ["green" for i in range(boxes1["prediction"].shape[0])]
    labels1 = pred1["labels"].detach().cpu()
    print(labels)
    result = draw_bounding_boxes(
        image=pil_to_tensor(image).to(torch.uint8),
        boxes=boxes1["prediction"],
        colors=colors1,
        labels=[str(model.config.id2label[label]) for label in labels1.tolist()])
    img = Image.fromarray(result.permute(1, 2, 0).numpy())
    img.save(f"{Path(__file__).parent.absolute()}/../images/test/testtabletransformerinference_{num}_1.jpg")

if __name__=='__main__':
    #inferencetest(f"{Path(__file__).parent.absolute()}/testdata/demotable_table_transformer.png",num= 0)
    #inferencetest(f"{Path(__file__).parent.absolute()}/../data/BonnData/test/I_HA_Rep_89_Nr_16160_0170/I_HA_Rep_89_Nr_16160_0170_table_0.jpg", num=1)
    inferencetest2(modelpath=f"{Path(__file__).parent.absolute()}/../checkpoints/tabletransformer/tabletransformer_v1.1-all_BonnDataFullImage_tabletransformer_v1.1test_BonnData_fullimage_e2_valid_end.pt",
        imgpath=f"{Path(__file__).parent.absolute()}/../data/BonnData/test/I_HA_Rep_89_Nr_16160_0170/I_HA_Rep_89_Nr_16160_0170.jpg",
        num=15)
    inferencetest2(
        modelpath=f"{Path(__file__).parent.absolute()}/../checkpoints/tabletransformer/tabletransformer_v1.1-all_BonnDataFullImage_tabletransformer_v1.1test_BonnData_fullimage_e2_valid_end.pt",
        imgpath=f"{Path(__file__).parent.absolute()}/testdata/demotable_table_transformer.png",
        num=30)
    inferencetest2(
        imgpath=f"{Path(__file__).parent.absolute()}/../data/BonnData/test/I_HA_Rep_89_Nr_16160_0170/I_HA_Rep_89_Nr_16160_0170.jpg",
        num=50)
    #inferencetest(
    #    modelpath=f"{Path(__file__).parent.absolute()}/../checkpoints/tabletransformer/tabletransformer_v1.1-all_BonnDataFullImage_tabletransformer_v1.1_BonnData_fullimage_e250_valid_end.ckpt",
    #    imgpath=f"{Path(__file__).parent.absolute()}/../data/BonnData/test/I_HA_Rep_89_Nr_16160_0170/I_HA_Rep_89_Nr_16160_0170.jpg",
    #    num=3)
    #inferencetest(
    #    modelpath=f"{Path(__file__).parent.absolute()}/../tests/tableformer/1",
    #    imgpath=f"{Path(__file__).parent.absolute()}/../data/BonnData/test/I_HA_Rep_89_Nr_16160_0170/I_HA_Rep_89_Nr_16160_0170.jpg",
    #    num=4)