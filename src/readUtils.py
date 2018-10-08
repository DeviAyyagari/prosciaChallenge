from os import listdir
from os.path import isdir, isfile, join
from math import floor
import matplotlib.image as mpimg       # reading images to numpy arrays
from sklearn.model_selection import train_test_split

def readFileStructure(inputDir):
  """
    Assume a directory structure like
    inputDir/
        - type1
            -- name1.png
            ...
            -- namen.png
        - type2
            -- name1.png
            ...
            -- namen.png
        ...
        - typeN
            -- name1.png
            ...
            -- namen.png
    inputDir/<typei>/*.png
    """
  types = (dir for dir in listdir(inputDir) if isdir(join(inputDir, dir)))
  fileTypeDict = {}
  for typ in types:
    typePath = join(inputDir, typ)
    fileTypeDict[typ] = [join(typePath, f) for f in listdir(typePath) if isfile(join(typePath, f))]

  return fileTypeDict

def imgToNumpyArray(imgPath):
    return mpimg.imread(imgPath)

def imgListToNumpyArrayList(imgPaths):
    data = []
    for path in imgPaths:
        data.append(imgToNumpyArray(path))
    return data

def classifyTestAndTrainPixelData(metaData, trainFraction):
    """
    This function takes structure of the data(dict of Tissuetype and list of files as values)
    reads the pixel data for the files and splits the data, for each "type" as test and
    train by the given "trainSize" fraction.
    This is done to keep training data of certain size for each TissueType
    """
    xTrain = []
    xTest = []
    yTrain = []
    yTest = []
    for typeName, filePaths in metaData.items():
        y = [typeName]*len(filePaths)
        x = imgListToNumpyArrayList(filePaths)
        _xTrain, _xTest, _yTrain, _yTest = train_test_split(x, y, train_size=trainFraction)
        xTrain.extend(_xTrain)
        yTrain.extend(_yTrain)
        xTest.extend(_xTest)
        yTest.extend(_yTest)
    return xTrain, xTest, yTrain, yTest
