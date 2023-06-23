# Deep feature extraction

import numpy as np
import tensorflow as tf
import collections
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom
import os, sys
import pandas as pd
from keras.preprocessing import image

# Load model
from tensorflow.keras.applications.resnet50 import ResNet50 # pip install tensorflow
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import Model

## Set GPU
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

base_model = ResNet50(weights='imagenet', include_top=True)
model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

imgDir = r'L:\basic\divi\jstoker\slicer_pdac\Master Students SS 23\Matej\keras_data_new'
dirlist = os.listdir(imgDir)


def loadSegArraywithID(fold, iden):
    """
    Loads the segmented image array with the specified identifier from the given folder.

    Parameters:
        :param: fold (str): The path to the folder containing the segmented images.
        :param: iden (str): The identifier used to search for the desired segmented image.

    :return: sitk.Image: The segmented image loaded using SimpleITK.
    """
    path = fold
    pathList = os.listdir(path)
    segPath = [os.path.join(path, i) for i in pathList if ('seg' in i.lower()) & (iden in i.lower())][0]
    seg = sitk.ReadImage(segPath)
    return seg


def loadImgArraywithID(fold, iden):
    """
    Loads the image array with the specified identifier from the given folder.

    Parameters:
        :param: fold (str): The path to the folder containing the images.
        :param: iden (str): The identifier used to search for the desired image.

    :return: sitk.Image: The image loaded using SimpleITK.
    """
    path = fold
    pathList = os.listdir(path)

    imgPath = [os.path.join(path, i) for i in pathList if ('im' in i.lower()) & (iden in i.lower())][0]
    img = sitk.ReadImage(imgPath)
    return img


def maskcroppingbox(images_array, use2D=False):
    """
    Calculates the bounding box coordinates based on the non-zero elements in the images_array. Widenes the box
    along each axis.

    Parameters:
        :param: images_array (numpy.ndarray): The input array containing the image data.
        :param: use2D (bool, optional): Flag indicating whether to perform 2D cropping. Defaults to False.

    :return: tuple: A tuple of tuples representing the starting and ending coordinates of the cropping box
                    in the format ((zstart, ystart, xstart), (zstop, ystop, xstop)).
    """
    images_array_2 = np.argwhere(images_array)
    print(images_array_2.shape)
    (zstart, ystart, xstart), (zstop, ystop, xstop) = images_array_2.min(axis=0), images_array_2.max(axis=0) + 1
    print(zstart, ystart, xstart)
    print(zstop, ystop, xstop)

    # Modify the slicing indices to widen the ROI
    zstart -= 2  # Decrease the starting z index
    zstop += 2  # Increase the ending z index
    ystart -= 10  # Decrease the starting y index
    ystop += 10  # Increase the ending y index
    xstart -= 10  # Decrease the starting x index
    xstop += 10  # Increase the ending x index
    return (zstart, ystart, xstart), (zstop, ystop, xstop)


def featureextraction(imageFilepath, maskFilepath):
    """
    Performs feature extraction on the image within the specified ROI defined by the mask.

    Parameters:
        :param: imageFilepath (str): Filepath of the input image.
        :param: maskFilepath (str): Filepath of the mask defining the ROI.

    :return: collections.OrderedDict: Dictionary of extracted features.
    """
    image_array = sitk.GetArrayFromImage(imageFilepath)
    mask_array = sitk.GetArrayFromImage(maskFilepath)

    if image_array.ndim == 4:
        image_array = image_array[:, :, :, 0] # Remove the 4th dimension

    (zstart, ystart, xstart), (zstop, ystop, xstop) = maskcroppingbox(mask_array, use2D=False)
    roi_images = image_array[zstart - 1:zstop + 1, ystart:ystop, xstart:xstop].transpose((2, 1, 0))
    roi_images1 = zoom(roi_images, zoom=[224 / roi_images.shape[0], 224 / roi_images.shape[1], 1], order=3)
    roi_images2 = np.array(roi_images1, dtype=np.float)
    x = tf.keras.preprocessing.image.img_to_array(roi_images2)
    num = []
    for i in range(zstart, zstop):
        mask_array = np.array(mask_array, dtype='uint8')
        images_array_3 = mask_array[:, :, i]
        num1 = images_array_3.sum()
        num.append(num1)
    maxindex = num.index(max(num))
    print(max(num), ([*range(zstart, zstop)][num.index(max(num))]))
    x1 = np.asarray(x[:, :, maxindex - 1])
    x2 = np.asarray(x[:, :, maxindex])  # ??????slice
    x3 = np.asarray(x[:, :, maxindex + 1])
    print(x1.shape)
    x1 = np.expand_dims(x1, axis=0)
    x2 = np.expand_dims(x2, axis=0)
    x3 = np.expand_dims(x3, axis=0)
    a1 = np.asarray(x1)
    a2 = np.asarray(x2)
    a3 = np.asarray(x3)
    print(a1.shape)
    mylist = [a1, a2, a3]
    x = np.asarray(mylist)
    print(x.shape)
    x = np.transpose(x, (1, 2, 3, 0))
    print(x.shape)
    x = preprocess_input(x)

    base_model_pool_features = model.predict(x)

    features = base_model_pool_features[0]

    deeplearningfeatures = collections.OrderedDict()
    for ind_, f_ in enumerate(features):
        deeplearningfeatures[str(ind_)] = f_
    return deeplearningfeatures


featureDict = {}
for ind in range(len(dirlist)):
    try:
        path = os.path.join(imgDir,dirlist[ind])
        seg = loadSegArraywithID(path,'seg')
        im = loadImgArraywithID(path,'im')

        deeplearningfeatures = featureextraction(im,seg)

        result = deeplearningfeatures
        key = list(result.keys())
        key = key[0:]

        feature = []
        for jind in range(len(key)):
            feature.append(result[key[jind]])

        featureDict[dirlist[ind]] = feature
        dictkey = key
    except ValueError as e:
        # passed.append(path)
        print(featureDict[dirlist[ind]], "PASSED due to ValueError")
        print(e)
        pass
    except IndexError as ie:
        # passed.append(path)
        # print(featureDict[dirlist[ind]], "PASSED due to IndexError")
        print(ie)
        pass
    except RuntimeError as re:
        # passed.append(path)
        # print(featureDict[dirlist[ind]], "PASSED due to RuntimeError")
        print(re)
        pass

# dataframe = pd.DataFrame.from_dict(featureDict, orient='index', columns='dictkey')
dataframe = pd.DataFrame.from_dict(featureDict, orient='index')
dataframe.to_csv('./feature.csv')







