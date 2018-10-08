from sklearn.svm import SVC
from sklearn import metrics
from imageProcessing import processImages
import readUtils

def findAccuracy(yTest, yPred):
    return sum((a==b for a,b in zip(yTest, yPred)))/len(yPred)

def train(inputPath):
    bestAccuracy = -1
    bestModel = None
    bestYPred = None
    metaData = readUtils.readFileStructure(inputPath)
    trainSize = 0.75
    for i in range(100):
        xTrain, xTest, yTrain, yTest = readUtils.classifyTestAndTrainPixelData(metaData, trainSize)
        xTrain = processImages(xTrain)
        xTest = processImages(xTest)
        svclassifier = SVC(gamma=0.001, kernel='rbf')
        svclassifier.fit(xTrain, yTrain)
        yPred = svclassifier.predict(xTest)
        accuracy = findAccuracy(yTest, yPred)
        if(accuracy > bestAccuracy):
            bestAccuracy = accuracy
            bestYPred = yPred
            bestModel = svclassifier
    print("Accuracy of the model is:"+ str(bestAccuracy))
    print("Confusion Matrix for classifier {}\n{}".format(bestModel, metrics.confusion_matrix(yTest, bestYPred)))
    print("Classification report: {}".format(metrics.classification_report(yTest, bestYPred)))
    return bestModel
