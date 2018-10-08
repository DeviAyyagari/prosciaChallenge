# imports
import numpy as np                     # numeric python lib
from skimage.color import rgb2grey

def processImage(imgArr):
    return np.concatenate((imgArr.flatten(), rgb2grey(imgArr).flatten()))

def processImages(imgArr):
    for i in range(len(imgArr)):
        imgArr[i] = processImage(imgArr[i])
    return imgArr
