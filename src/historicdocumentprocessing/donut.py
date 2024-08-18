import json
import os
from pathlib import Path
import torch
from donut import DonutModel
from PIL import Image as image
import glob
from tqdm import tqdm


def testinference(model: DonutModel, img: image.Image, dataset: str, target: str,
                  targetloc: str = f"{Path(__file__).parent.absolute()}/../../results",
                  pretrained: str = "naver-clova-ix/donut-base"):
    #referenced https://towardsdatascience.com/ocr-free-document-understanding-with-donut-1acfbdf099be
    """ Tests Donut Model inference on singular image and saves result."""

    model.eval()
    output = model.inference(image=img, prompt=f"<s_synthdog>")
    #output = model.inference(image=img, prompt=f"<s_cord-v2>")
    saveloc = f"{targetloc}/{pretrained}/{dataset}"
    savefile = f"{targetloc}/{pretrained}/{dataset}/{target}.json"
    os.makedirs(saveloc, exist_ok=True)
    with open(savefile, "w") as out:
        out.write(json.dumps(output, indent=4))
    print(output)
    return


def testloop(pretrained: str = "naver-clova-ix/donut-base",
             imgloc: str = f"{Path(__file__).parent.absolute()}/../../data/EngNewspaper/raw"):
    model = DonutModel.from_pretrained(pretrained)
    if torch.cuda.is_available():
        model.half()
        device = torch.device("cuda")
        model.to(device)
    else:
        print("Cuda not available")
        return
    dataset = imgloc.split("/")[-1]
    imgs = glob.glob(f"{imgloc}/*.jpg")
    for impath in tqdm(imgs):
        target = impath.split("/")[-1].split(".")[-2]
        img = image.open(impath).convert("RGB")
        #print(dataset, target)
        testinference(model=model, img=img, dataset=dataset, target=target, pretrained=pretrained)


if __name__ == '__main__':
    testloop()
