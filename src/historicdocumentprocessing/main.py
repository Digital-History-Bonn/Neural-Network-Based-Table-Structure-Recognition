# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from donut import DonutModel
from PIL import Image as image
from matplotlib import pyplot as plt
from pathlib import Path
import torch
import datasets
import json
import os
from urllib import request
import cv2
import numpy as np
from typing import List
from tqdm import tqdm


def retrieveengnespaperimg(url:str, identifier:str, targetloc: str = f"{Path(__file__).parent.absolute()}/../../data/EngNewspaper"):
    """Retrieve the corresponding newspaper image from a given American Stories url und save it.

    Args:
        url (str) : the image url
        identifier (str) : image name (usually lcnn_issn_{nr in Dataset})
        targetloc (str) : save location"""

    os.makedirs(targetloc, exist_ok=True)
    #file = f"{targetloc}/{str(num)}.jp2"
    filejpg = f"{targetloc}/{identifier}.jpg"
    opener= request.build_opener()
    opener.addheaders = [('User-Agent', 'MyApp/1.0')]
    request.install_opener(opener)
    response = np.asarray(bytearray(request.urlopen(url).read()), "uint8")
    img = cv2.imdecode(response, cv2.IMREAD_COLOR)
    cv2.imwrite(filejpg, img)
    #request.urlretrieve(url, filejpg)
    #img = cv2.imread(file, cv2.IMREAD_COLOR)
    #plt.imsave(filejpg, img)
    return

def processengnewspaper(targetloc: str = f"{Path(__file__).parent.absolute()}/../../data/EngNewspaper", year: List[str] = ["1870"]):
    """Processes AmericanStories Dataset: find the entries where Chronicling America URL is available
    and save those and corresponding JSON as lcnn_issn_{nr in Dataset}.

    Args:
        targetloc (str) : target save location
        year (str)
    """
    #referenced https://colab.research.google.com/drive/1ifzTDNDtfrrTy-i7uaq3CALwIWa7GB9A to write code
    dataset = datasets.load_dataset("dell-research-harvard/AmericanStories",
                                    "subset_years_content_regions",
                                    year_list=year
                                    )
    #print(dataset['1870'][6])
    #sample = json.loads(dataset['1870'][6]['raw_data_string'])
    #print(sample)
    #print(sample['scan'].keys())
    for i in tqdm(range(100)):
        current = json.loads(dataset['1870'][i]['raw_data_string'])
        #print(current.keys())
        #print(current['page_number'])
        #print(i, current['scan'].keys())
        #print(current)
        if 'jp2_url' in current['scan'].keys() and 'lccn' in current.keys():
            #print(sample['scan']['jp2_url'])
            lccn = current['lccn']['lccn']
            issn = current['lccn']['issn']
            identifier = f"{lccn}_{issn}_{str(i)}"
            #print(type(current))
            currentjson = json.dumps(current, indent=4)
            with open(f"{targetloc}/{identifier}.json", "w") as out:
                out.write(currentjson)
            retrieveengnespaperimg(current['scan']['jp2_url'],identifier)
    #    sampledict = json.loads(dataset['1870'][i])


def testrun(impath: str):
    #referenced https://towardsdatascience.com/ocr-free-document-understanding-with-donut-1acfbdf099be
    model = DonutModel.from_pretrained("naver-clova-ix/donut-base")
    img = image.open(impath).convert("RGB")
    #plt.imshow(img)
    #plt.show()
    if torch.cuda.is_available():
        model.half()
        device = torch.device("cuda")
        model.to(device)
    else:
        print("Cuda not available")
        return
    model.eval()
    output = model.inference(image=img, prompt=f"<s_synthdog>")
    #output = model.inference(image=img, prompt=f"<s_cord-v2>")
    print(output)
    return



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('PyCharm')
    #print(f"{Path(__file__).parent.absolute()}/../../data/BonnData/test/Koelnische_Zeitung_1866-06_1866-09_0046.jpg")
    #testrun(f"{Path(__file__).parent.absolute()}/../../data/BonnData/test/Koelnische_Zeitung_1866-06_1866-09_0046.jpg")
    #testrun(f"{Path(__file__).parent.absolute()}/../../data/BonnData/test/zeitungsminicrop.jpg")
    processengnewspaper()
    #testrun(f"{Path(__file__).parent.absolute()}/../../data/EngNewspaper/6.jpg")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
