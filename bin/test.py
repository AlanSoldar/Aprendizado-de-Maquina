#!/usr/bin/python3

import pandas as pd
from data import *
from analyzer import *
from knn import *

#Tests to data.py module
# That module contains all method necessary to filter and create stratified folds from original csv
test = Data("data\diabetes.csv",5)

print(test.getQtyOfInstances())
print("***************************************************")
print(test.getDataByFold())
print("***************************************************")
print(test.getRawData())
print("***************************************************")
print("Percent by class: Raw",test.getPercentOfClasses(test.getRawData(),'Potability'))
i=1
dataByFold = test.getDataByFold()
for key in dataByFold:
    #print(key)
    #print(dataByFold[key])
    print("Percent by class: Fold ",key," - ",test.getPercentOfClasses(dataByFold[key],'Potability'))
    i += 1

#Test to analyzer.py
# That module contains all methods necessary to analyse all data got from KNN algorithm executions
# all functions for analysis are based on Confusion Matrix
#Classe positiva = 1 e Classe negativa = 0, lista precisa estar nesta ordem
testAnalysis = Analyzer([1,0])
testAnalysis.addValueInConfusionMatrix(0,0)
testAnalysis.addValueInConfusionMatrix(1,1)
testAnalysis.addValueInConfusionMatrix(0,1)
testAnalysis.addValueInConfusionMatrix(0,0)
testAnalysis.addValueInConfusionMatrix(1,1)
testAnalysis.addValueInConfusionMatrix(0,0)
testAnalysis.addValueInConfusionMatrix(0,1)
testAnalysis.addValueInConfusionMatrix(0,0)
testAnalysis.addValueInConfusionMatrix(0,1)
testAnalysis.addValueInConfusionMatrix(1,1)
testAnalysis.addValueInConfusionMatrix(1,0)
testAnalysis.addValueInConfusionMatrix(1,0)
testAnalysis.addValueInConfusionMatrix(0,0)
testAnalysis.addValueInConfusionMatrix(1,0)
testAnalysis.addValueInConfusionMatrix(1,1)
testAnalysis.addValueInConfusionMatrix(1,0)

print("***************************************************")
print(testAnalysis.getConfusionMatrix())
print("***************************************************")
print(testAnalysis.calcAccuracy())
print("***************************************************")
print(testAnalysis.calcRecall(1))
print("***************************************************")
print(testAnalysis.calcPrecision(1))
print("***************************************************")
print(testAnalysis.calcFBethaMeasure(1))
print("***************************************************")


#Tests to Knn.py module
# That module contains all methods necessary to implement KNN algorithm
data = test.getDataByFold()

for keyFoldTest in data:
    foldTrainingSet = pd.DataFrame()
    foldTestSet = pd.DataFrame()
    for keyFoldTraining in data:
        if (keyFoldTest != keyFoldTraining):
            foldTrainingSet = foldTrainingSet.append(data[keyFoldTraining], ignore_index=True)
        else:
            foldTestSet = foldTestSet.append(data[keyFoldTest], ignore_index=True)

    testKNN = Knn(foldTrainingSet)
    result = testKNN.knnAlgorithm(foldTestSet,5)
    print("***************************************************")
    print(result)
    print("***************************************************")