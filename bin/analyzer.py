#!/usr/bin/python

import pandas as pd
import math
import numpy

class Analyzer:

    def __init__(self,listKeyClasses):
        self.__listKeysClasses = listKeyClasses
        self.__createConfusionMatrix(listKeyClasses)

    def calcAccuracy(self):
        sumTrues = 0.0
        sumFalses = 0.0
        for keyPredicted in self.__listKeysClasses:
            sumTrues += self.__confusionMatrix[keyPredicted].loc[keyPredicted]
            for keyOriginal in self.__listKeysClasses:
                if(keyPredicted != keyOriginal):
                    sumFalses += self.__confusionMatrix[keyPredicted].loc[keyOriginal]
        n = sumTrues + sumFalses
        print(n)

        return sumTrues/n

    def calcRecall(self,positiveClassOriginal):
        sumClassOriginal = 0.0
        for keyPredicted in self.__listKeysClasses:
            sumClassOriginal += self.__confusionMatrix[keyPredicted].loc[positiveClassOriginal]

        return self.__confusionMatrix[positiveClassOriginal].loc[positiveClassOriginal] / sumClassOriginal

    def calcPrecision(self,positiveClassPredicted):
        sumClassPredicted = 0.0
        for keyOriginal in self.__listKeysClasses:
            sumClassPredicted += self.__confusionMatrix[positiveClassPredicted].loc[keyOriginal]

        return self.__confusionMatrix[positiveClassPredicted].loc[positiveClassPredicted] / sumClassPredicted

    def calcFBethaMeasure(self, betha):
        num = self.calcPrecision(self.__listKeysClasses[0]) * self.calcRecall(self.__listKeysClasses[0])
        denom = (pow(betha,2)*self.calcPrecision(self.__listKeysClasses[0])) + self.calcRecall(self.__listKeysClasses[0])
        return (1+pow(betha,2))*((num)/(denom))

    def calcAverage(self,valuesList):
        return numpy.mean(valuesList)

    def calcStandarDeviation(self,valuesList):
        return numpy.std(valuesList)

    def __createConfusionMatrix(self,listKeyClasses):
        self.__confusionMatrix = pd.DataFrame(numpy.zeros(shape=(len(listKeyClasses),len(listKeyClasses))), index=listKeyClasses,columns=listKeyClasses)
        
    def addValueInConfusionMatrix(self,keyClassPredicted, keyClassOriginal):
        if(self.__confusionMatrix[keyClassPredicted].loc[keyClassOriginal] > 0.0):
            self.__confusionMatrix[keyClassPredicted].loc[keyClassOriginal] += 1.0
        else:
            self.__confusionMatrix[keyClassPredicted].loc[keyClassOriginal] = 1.0

    def getConfusionMatrix(self):
        return self.__confusionMatrix