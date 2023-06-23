import numpy as np
import collections
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom
import os,sys
import pandas as pd
import radiomics
from radiomics import featureextractor
import openpyxl


imgDir = r"L:\basic\divi\jstoker\slicer_pdac\Master Students SS 23\Matej\keras_data_new"
dirlist = os.listdir(imgDir)


def loadSegArraywithID(fold, iden):
    """
    Load segmentations in NRRD format with a specific identifier.

    Parameters:
        :param: fold (str): Folder path containing the segmentation files.
        :param: iden (str): Identifier to filter the segmentation files.

    :return: seg (sitk.Image): The loaded segmentation image.
    :raise: IndexError: If no segmentation file is found with the specified identifier.

    """
    path = fold
    pathList = os.listdir(path)

    segPath = [os.path.join(path,i) for i in pathList if ('seg' in i.lower()) & (iden in i.lower())][0]
    seg = sitk.ReadImage(segPath)
    return seg


def loadImgArraywithID(fold, iden):
    """
    Load images in NIfTI or NRRD format with a specific identifier.

    Parameters:
        :param: fold (str): Folder path containing the image files.
        :param: iden (str): Identifier to filter the image files.

    :return: img (sitk.Image): The loaded image.

    """
    path = fold
    pathList = os.listdir(path)

    imgPath = [os.path.join(path,i) for i in pathList if ('im' in i.lower()) & (iden in i.lower())][0]
    img = sitk.ReadImage(imgPath)
    # print(img.GetPixelIDValue())
    if img.GetPixelIDValue() == 3:
        pass
    elif img.GetPixelIDValue() == 2 or img.GetPixelIDValue() == 9:
        img = sitk.Cast(img, sitk.sitkUInt8)
    elif img.GetPixelIDValue() == 15:
        img = sitk.VectorIndexSelectionCast(img, sitk.sitkUInt8, 0)
    elif img.GetPixelIDValue() == 11:
        img = img
    return img

# Feature Extraction
featureDict = {}
passed = []

for ind in range(len(dirlist)):
    path = os.path.join(imgDir,dirlist[ind])
    print(path)

    try: # you can make your own pipeline to import data, but it must be SimpleITK images
        mask = loadSegArraywithID(path,'seg')  # see line 26 !
        img = loadImgArraywithID(path,'im')      # see line 35 !
        params = r"L:\basic\divi\jstoker\slicer_pdac\Master Students SS 23\Matej\Code\tumor.yaml"

        extractor = featureextractor.RadiomicsFeatureExtractor(params)
        extractor.settings["geometryTolerance"] = 1

        result = extractor.execute(img,mask)
        key = list(result.keys())
        key = key[1:]

        feature = []
        for jind in range(len(key)):
            feature.append(result[key[jind]])

        featureDict[dirlist[ind]] = feature
        dictkey = key

    except RuntimeError as re:
        passed.append(path)
        print(f"{path} passed due to RE; ", re)

dataframe = pd.DataFrame.from_dict(featureDict, orient='index', columns=dictkey)
dataframe.to_excel(r"L:\basic\divi\jstoker\slicer_pdac\Master Students SS 23\Matej\keras_data_OUTPUT\radiomics_2.xlsx", header = True, index = True)