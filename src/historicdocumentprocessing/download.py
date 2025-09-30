"""Script to download datasets."""

import os
import zipfile
from pathlib import Path
from urllib import request


def download_glosat() -> None:
    """Download script for GloSAT dataset."""
    print("process can take a quite some time")
    print("downloading...")

    # define paths and create folder
    url = "https://zenodo.org/records/5363457/files/datasets.zip"
    target = f"{Path(__file__).parent.absolute()}/../../data/GloSAT/"
    file = f"{target}/datasets.zip"
    os.makedirs(target, exist_ok=True)

    # download dataset
    request.urlretrieve(url, file)

    # extract zip file
    print("extracting...")
    with zipfile.ZipFile(file, "r") as zipper:
        folder = Path(target)
        folder.mkdir(exist_ok=True)
        zipper.extractall(path=folder)

    # remove zip file
    os.remove(file)


def download_ours() -> None:
    """Download script for our data."""
    # TODO: implement download script for our dataset
    pass


def main(dataset: str = "GloSAT") -> None:
    """Download script for datasets. 'ours' currently not implemented.

    Args:
        dataset: name of the dataset ('GloSAT' or 'ours')

    Raises:
        Exception: if there is no download script for the given dataset
    """
    if dataset.lower() == "glosat":
        download_glosat()
    elif dataset.lower() == "ours":
        download_ours()

    else:
        raise Exception("No download script for this dataset!")


if __name__ == "__main__":
    main("GloSAT")
