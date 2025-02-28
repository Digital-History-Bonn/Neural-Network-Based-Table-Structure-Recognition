"""Split test."""
import glob
from pathlib import Path

import pandas as pd


def test_validsplitsanity():
    """Assert that actual valid split is identical to the one noted in file."""
    validpath = f"{Path(__file__).parent.absolute()}/../data/Tablesinthewild"
    validfile = f"{Path(__file__).parent.absolute()}/../data/Tablesinthewild/split.json"
    validfiles = [v.split("/")[-1] for v in glob.glob(f"{validpath}/valid/*")]
    imnames = pd.read_json(validfile)["validation"]
    validsaved = [im for im in imnames]
    assert set(validfiles) == set(validsaved)


if __name__ == "__main__":
    test_validsplitsanity()
