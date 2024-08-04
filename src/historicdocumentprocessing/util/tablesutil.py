from pathlib import Path

import pandas as pd


def BonnTablebyCat(categoryfile: str = f"{Path(__file__).parent.absolute()}/../../../data/BonnData/Tabellen/allinfosubset_manuell_vervollständigt.xlsx",
                   resultfile: str = f"{Path(__file__).parent.absolute()}/../../../results/kosmos25/testevaltotal/BonnData_Tables/fullimageiodt.csv", resultmetric: str = "iodt"):
    """
    filter bonntable eval results by category and calculate wf1 over different categories
    Args:
        categoryfile:
        resultfile:

    Returns:

    """
    df = pd.read_csv(resultfile)
    catinfo = pd.read_excel(categoryfile)
    df = df.rename(columns={'img': 'Dateiname'})
    df1 = pd.merge(df, catinfo, on='Dateiname')
    subsetwf1df = {'wf1': [], 'category': [], 'len': []}
    #replace category 1 by 2 since there are no images without tables in test dataset
    df1 = df1.replace({'category':1},2)
    #print(df1.category, df1.Dateiname)
    for cat in df1.category.unique():
        if not pd.isna(cat):
            subset = df1[df1.category == cat]
            subsetwf1df['wf1'].append(subset.wf1.sum() / len(subset))
            subsetwf1df['category'].append(cat)
            subsetwf1df['len'].append(len(subset))
    if len(df1[pd.isna(df1.category)]) > 0:
        subset = df1[pd.isna(df1.category)]
        subsetwf1df['category'].append('no category')
        subsetwf1df['len'].append(len(subset))
        subsetwf1df['wf1'].append(subset.wf1.sum() / len(subset))
    #print(subsetwf1df)
    saveloc = f"{'/'.join(resultfile.split('/')[:-1])}/{resultmetric}_bycategory.xlsx"
    pd.DataFrame(subsetwf1df).set_index('category').to_excel(saveloc)

if __name__ == '__main__':
    BonnTablebyCat()
    BonnTablebyCat(resultfile=f"{Path(__file__).parent.absolute()}/../../../results/kosmos25/testevaltotal/BonnData_Tables/fullimageiou.csv", resultmetric="iou")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/tableareacutout/BonnData/projectgroupmodelinference/iodt.csv")
    BonnTablebyCat(resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/tableareacutout/BonnData/projectgroupmodelinference/iou.csv", resultmetric="iou")
    BonnTablebyCat(resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/fullimg/BonnData/BonnDataFullImage1_BonnData_fullimage_e250_es.pt/tableareaonly/iou_0.5_0.9/fullimageiodt.csv")
    BonnTablebyCat(resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/fullimg/BonnData/BonnDataFullImage1_BonnData_fullimage_e250_es.pt/iou_0.5_0.9/fullimageiodt.csv")
    BonnTablebyCat(resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/fullimg/BonnData/BonnDataFullImage1_BonnData_fullimage_e250_es.pt/tableareaonly/iou_0.5_0.9/fullimageiou.csv", resultmetric="iou")
    BonnTablebyCat(resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/fullimg/BonnData/BonnDataFullImage1_BonnData_fullimage_e250_es.pt/iou_0.5_0.9/fullimageiou.csv", resultmetric="iou")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_es.pt/tableareaonly/iou_0.5_0.9/fullimageiodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_es.pt/iou_0.5_0.9/fullimageiodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_es.pt/tableareaonly/iou_0.5_0.9/fullimageiou.csv",
        resultmetric="iou")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_es.pt/iou_0.5_0.9/fullimageiou.csv",
        resultmetric="iou")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_end.pt/tableareaonly/iou_0.5_0.9/fullimageiodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_end.pt/iou_0.5_0.9/fullimageiodt.csv")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_end.pt/tableareaonly/iou_0.5_0.9/fullimageiou.csv",
        resultmetric="iou")
    BonnTablebyCat(
        resultfile=f"{Path(__file__).parent.absolute()}/../../../results/fasterrcnn/testeval/fullimg/BonnData/BonnDataFullImage_pretrain_GloSatFullImage1_GloSat_fullimage_e250_es_BonnData_fullimage_e250_end.pt/iou_0.5_0.9/fullimageiou.csv",
        resultmetric="iou")

