import glob
import os
import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def recreatetablesplit(datapath: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/preprocessed",
                       csvpath: str = f"{Path(__file__).parent.absolute()}/../../data/BonnData/Tabellen/run3_BonnData_cell_aug_loadrun_GloSAT_cell_aug_e250_es_e250_es.csv"):
    """
    Recreate the Test Split of the Bonn Table Dataset from the csv result file
    Args:
        datapath: path to preprocessed Bonn Table Data
        csvpath: path to Bonn Table csv result file

    Returns:
    """
    imnames = pd.read_csv(csvpath)["image_number"][1:-1]
    savelocs = "/".join(datapath.split("/")[:-1]) + "/test"
    os.makedirs(savelocs, exist_ok=True)
    #print(len(imnames))
    for i, imname in tqdm(enumerate(imnames)):
        imfolder = glob.glob(f"{datapath}/{str(imname)}")
        saveloc = f"{savelocs}/{imname}"
        dest = shutil.copytree(src=imfolder[0], dst=saveloc, dirs_exist_ok=True)
        #print(i)


if __name__ == '__main__':
    recreatetablesplit()
