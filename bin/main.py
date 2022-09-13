#!/usr/bin/python

import sys
import pandas as pd
from data import Data
from analyzer import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

table = Data("..\data\water_potability_reduced.csv",10)

analysis = Analyzer([1,0])
accuracy = []
precision = []
recall = []
f1Measure = []

data = table.getDataByFold()

print("|************************|************************|***********************|")
print("| Fold de Teste/Iteracao |        Acuracia        |      F1-measure       |")
print("|________________________|________________________|_______________________|")

for keyFoldTest in data:
    foldTrainingSet = pd.DataFrame()
    foldTestSet = pd.DataFrame()
    for keyFoldTraining in data:
        if (keyFoldTest != keyFoldTraining):
            foldTrainingSet = pd.concat([foldTrainingSet, data[keyFoldTraining]], ignore_index=True)
        else:
            foldTestSet = pd.concat([foldTestSet, data[keyFoldTest]], ignore_index=True)

    expectedTraining = foldTrainingSet['Potability'].to_numpy()
    expectedTest = foldTestSet['Potability'].to_numpy()
    
    mode = sys.argv[1]
    if mode == "KNN":
        print("Running KNN")
        test = KNeighborsClassifier(5)
    elif mode == "naive":
        print("Running naive bayes")
        test = GaussianNB()
    else:
        print("Running random forest")
        test = RandomForestClassifier()

    test.fit(foldTrainingSet.to_numpy(), expectedTraining)
    pred = test.predict(foldTestSet.to_numpy())
    print(pred)
    print(expectedTest)

    #calculo de metricas usando biblioteca
    # print("accuracy = ", metrics.accuracy_score(expectedTest, pred))
    # print("recall = ", metrics.recall_score(expectedTest, pred))
    # print("precision = ", metrics.precision_score(expectedTest, pred))
    # print("f1 = ", metrics.f1_score(expectedTest, pred))

    for index in range(len(pred)):
        analysis.addValueInConfusionMatrix(pred[index], expectedTest[index])

    #print(analysis.getConfusionMatrix())

    #calculo de metricas usando codigo implementado
    print("accuracy = ", analysis.calcAccuracy())
    print("precision = ", analysis.calcPrecision(1))
    print("recall = ", analysis.calcRecall(1))
    print("f1 = ", analysis.calcFBethaMeasure(1))


    