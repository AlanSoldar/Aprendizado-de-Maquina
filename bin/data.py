#!/usr/bin/python

import pandas as pd
import math

class Data:

    # variables for use in class = type
    #__proportions (dictionary)
    #__totalOfInstances (int)
    #__data (pandas DataFrame)
    #__dataNormalized (pandas DataFrame)
    #__dataByFold (pandas DataFrame)

    def __init__(self,filePath,numKFold):
        self.__readCsvFile(filePath)
        self.__normalizeAttributesValues()
        self.__setQtyOfInstances(self.__data)
        self.__divideInstancesInFolds(numKFold)
        
    def __readCsvFile(self,filePath):
        self.__data = pd.read_csv(filePath)

    def __normalizeAttributesValues(self):
        self.__dataNormalized = pd.DataFrame()
        for colName in self.__data.columns:
            if(colName != "Potability"):
                max = self.__getMaxValueByColumn(colName)
                min = self.__getMinValueByColumn(colName)
                i = 0
                newValList = []
                for cellValue in self.__data[colName]:
                    newValList.append((cellValue - min)/(max-min))
                self.__dataNormalized[colName] = pd.Series(newValList)
            else:
                self.__dataNormalized[colName] = self.__data[colName]
       
    def __getMaxValueByColumn(self,column):
        return max(self.__data[column])

    def __getMinValueByColumn(self,column):
        return min(self.__data[column])

    def __setQtyOfInstances(self, data):
        self.__totalOfInstances = len(data.index)

    def getQtyOfInstances(self):
        return self.__totalOfInstances

    def __divideInstancesInFolds(self,k):
        self.__calcProportions('Potability')
        keyList = self.__proportions.keys()
        numInstanceKeepProportion = {}
        restOfNumInstanceKeepProportion = {}

        qtdInstancesByFold = self.__totalOfInstances // k
        rest = self.__totalOfInstances % k

        if(rest > 0):
            auxQtdInstance = qtdInstancesByFold + 1
            rest -= 1
        else:
            auxQtdInstance = qtdInstancesByFold

        for key in keyList:
            numInstanceKeepProportion[key] = math.floor(auxQtdInstance*(self.__proportions[key]/self.__totalOfInstances))
            restOfNumInstanceKeepProportion[key] = self.__proportions[key] - (k*numInstanceKeepProportion[key])  

        self.__setInstancesInFolds(k,keyList,numInstanceKeepProportion,restOfNumInstanceKeepProportion,auxQtdInstance,rest,qtdInstancesByFold)
      
    def __calcProportions(self,column):
        self.__proportions = {}

        for cellValue in self.__dataNormalized[column]:
            if(cellValue in self.__proportions):
                self.__proportions[cellValue] += 1
            else:
                self.__proportions[cellValue] = 1

    def __setInstancesInFolds(self,k,keyList,numInstanceKeepProportion,restOfNumInstanceKeepProportion,auxQtdInstance,rest,qtdInstancesByFold):
        self.__dataByFold = {}
        dataTemp = self.__dataNormalized

        for i in range(k):
            auxOfProportion = {}
            #sum = 0
            for key in keyList:
                auxOfProportion[key] = numInstanceKeepProportion[key]
                if(restOfNumInstanceKeepProportion[key] > 0):
                   auxOfProportion[key] += 1
                   restOfNumInstanceKeepProportion[key] -= 1
            #    sum += auxOfProportion[key]

            #print("SumProportion: ",sum," QtdIntancesByFold: ",auxQtdInstance)

            foldTemp = pd.DataFrame()
            j = 0
            while(auxQtdInstance > 0):
                self.__setQtyOfInstances(dataTemp)
                if(j < self.__totalOfInstances):            
                    rowOfInstance = dataTemp.iloc[j]
                    if(auxOfProportion[rowOfInstance['Potability']] > 0):
                        foldTemp = foldTemp.append(rowOfInstance)
                        auxOfProportion[rowOfInstance['Potability']] -= 1
                        auxQtdInstance -= 1
                        dataTemp.drop([dataTemp.index[j]], inplace=True)
                else:
                    j = -1
                    #for key in keyList:
                    #    print(key,": ",auxOfProportion[key])
                    #print("Instances faltando: ", auxQtdInstance)
                    #print("Table: ", dataTemp)
                j += 1
            
            self.__dataByFold[i] = foldTemp

            if(rest > 0):
                auxQtdInstance = qtdInstancesByFold + 1
                rest -= 1
            else:
                auxQtdInstance = qtdInstancesByFold

    def getRawData(self):
        return self.__data

    def getDataByFold(self):
        return self.__dataByFold

    def getPercentOfClasses(self,data,column):
        auxPercent = {}
        countQtdOfInstances = 0
        percentValuesByClass = {}

        for cellValue in data[column]:
            countQtdOfInstances += 1
            if(cellValue in auxPercent):
                auxPercent[cellValue] += 1
            else:
                auxPercent[cellValue] = 1
        
        for key in auxPercent.keys():
            percentValuesByClass[key] = (auxPercent[key]/countQtdOfInstances)*100.0

        return percentValuesByClass

"""         
    def saveStatisticsInTableResult(self):
"""