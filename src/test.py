from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy
from imageProcessing import processImage
import readUtils

def image_type(model, image):
    if type(image) is str:
        imageArr = readUtils.imgToNumpyArray(image)
    elif type(image) is numpy.ndarray:
        imageArr = image
    else:
        print("Unrecognized image format. Give a path to image or image data as numpy array")
        return None
    imageArr = processImage(imageArr)
    predictedType = model.predict([imageArr])
    return predictedType[0]
