import json
import os
from pathlib import Path
from typing import List
from urllib import request

import cv2
import datasets
import numpy as np
from tqdm import tqdm


def retrieveengnespaperimg(
    url: str,
    identifier: str,
    targetloc: str = f"{Path(__file__).parent.absolute()}/../../data/EngNewspaper/unsorted",
    imresize: float = 0.4,
):
    """Retrieve the corresponding newspaper image from a given American Stories url und save it.

    Args:
        url (str) : the image url
        identifier (str) : image name (usually lcnn_issn_{nr in Dataset})
        targetloc (str) : save location
        imresize: image scale factor"""

    os.makedirs(targetloc, exist_ok=True)
    filejpg = f"{targetloc}/{identifier}.jpg"
    opener = request.build_opener()
    opener.addheaders = [("User-Agent", "MyApp/1.0")]
    request.install_opener(opener)
    response = np.asarray(bytearray(request.urlopen(url).read()), "uint8")
    img = cv2.imdecode(response, cv2.IMREAD_COLOR)
    cv2.imwrite(filejpg, cv2.resize(img, (0, 0), fx=imresize, fy=imresize))
    return


def processengnewspaper(
    targetloc: str = f"{Path(__file__).parent.absolute()}/../../data/EngNewspaper/unsorted",
    year: List[str] = ["1870"],
    imresize: float = 0.4,
):
    """Processes AmericanStories Dataset: find the entries where Chronicling America URL is available
    and save those and corresponding JSON as lcnn_issn_{nr in Dataset}.

    Args:
        imresize: image scale factor (!bounding box coords not yet resized accordingly!)
        targetloc (str) : target save location
        year (str)
    """
    # referenced https://colab.research.google.com/drive/1ifzTDNDtfrrTy-i7uaq3CALwIWa7GB9A to write code
    dataset = datasets.load_dataset(
        "dell-research-harvard/AmericanStories",
        "subset_years_content_regions",
        year_list=year,
    )

    os.makedirs(targetloc, exist_ok=True)
    # print(len(dataset['1870']))
    for i in tqdm(range(5000, 10000)):
        current = json.loads(dataset["1870"][i]["raw_data_string"])
        if (
            "jp2_url" in current["scan"].keys()
            and "lccn" in current.keys()
            and len(current["bboxes"]) > 0
        ):
            lccn = current["lccn"]["lccn"]
            issn = current["lccn"]["issn"]
            # identifier = f"test{i}"
            if imresize != 0:
                current["scan"].update({"rescalefactor": imresize})
            identifier = f"{lccn}_{issn}_{str(i)}"
            currentjson = json.dumps(current, indent=4)
            with open(f"{targetloc}/{identifier}.json", "w") as out:
                out.write(currentjson)
            retrieveengnespaperimg(
                current["scan"]["jp2_url"], identifier, imresize=imresize
            )


if __name__ == "__main__":
    processengnewspaper()
