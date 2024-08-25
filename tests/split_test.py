import glob
from pathlib import Path

import pandas as pd


def validsplitsanitytest(validpath:str=f"{Path(__file__).parent.absolute()}/../data/Tablesinthewild", validfile:str=f"{Path(__file__).parent.absolute()}/../data/Tablesinthewild/split.json"):
    """assert that actual valid split is identical to the one noted in file"""
    validfiles = [v.split("/")[-1] for v in glob.glob(f"{validpath}/valid/*")]
    imnames = pd.read_json(validfile)["validation"]
    validsaved = [im for im in imnames]
    assert set(validfiles)==set(validsaved)

if __name__=='__main__':
    validsplitsanitytest()