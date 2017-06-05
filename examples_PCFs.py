import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import time
import PolyhedralConicFunctions as pc

"""



    # Polyhedral Conic Functions Example
# PCF algorithm requires both data clusters
# This function seperates data to clusters according to their labels
def seperatetoAB(data, labels, indexes):
    A = []
    B = []
    for i in indexes:
        if labels[i] == -1:
            A.append(data[i])
        elif labels[i] == 1:
            B.append(data[i])
    A = np.array(A)
    B = np.array(B)
    return A, B


# This function converts given labels (l1,l2) to 1, -1
def convertLabels(labelData, l1, l2):
    lbls = np.zeros(len(labelData))
    for i in range(len(labelData)):
        if labelData[i] == l1:
            lbls[i] = -1
        elif labelData[i] == l2:
            lbls[i] = 1
    return lbls

finalTest =[]
finalTRain=[]
finalStd=[]
finalTime=[]
finalPcfs=[]
kats= np.linspace(1,1,10)
for kt in kats:
        f = open('PimaDiabetes.txt')
        X = []
        labels = []
        for line in f:
            row = []
            line = line.split(',')
            labels.append(int(line[-1]))
            X.append([float(line[i]) for i in range(len(line)-1)])
        labels = np.array(labels)
        X = np.array(X)
        acc = []
        timeS = []
        timeF = []
        labels = convertLabels(labels, 0, 1)
        skf = StratifiedKFold(n_splits=10)
        accTrain = []
        pcfNumber=[]
        for train, test in skf.split(X,labels):
            sepData = seperatetoAB(X, labels, train)
            timeS.append(time.time())
            pModel = pc.PCF_iterative()
            pModel.fit(sepData[0], sepData[1])
            timeF.append(time.time())
            pcfNumber.append(len( pModel.pcfs))
            acc.append(accuracy_score(labels[test], pModel.predict(X[test],-1,1)))
            accTrain.append(accuracy_score(labels[train],pModel.predict(X[train],-1,1)))
        print "Training Acc:", sum(accTrain)/len(accTrain)
        print "Test Acc:", sum(acc)/len(acc)
        print "Std Acc", np.std(acc)
        print "Training Time",sum([timeF[i]-timeS[i] for i in range(len(timeS))])/len(timeS)
        print "**************************************************kt: ",kt
        print "Training Acc: ", sum(accTrain)/len(accTrain)
        finalTRain.append(np.mean(accTrain))
        print '\033[1m' + 'Test Accuracy'+str((sum(acc)/ len(acc))) + '\033[0m'
        finalTest.append(np.mean(acc))
        print "std Acc: ", np.std(acc)
        finalStd.append(np.std(acc))
        print "Training Time",sum([timeF[i]-timeS[i] for i in range(len(timeS))])/len(timeS)
        finalTime.append(sum([timeF[i]-timeS[i] for i in range(len(timeS))])/len(timeS))
        print "Av. Number PCF",np.mean(pcfNumber)
        finalPcfs.append(np.mean(pcfNumber))
print "---------------------Final---------------kt: ",kt
print "Training", np.mean(finalTRain)
print "Test",np.mean(finalTest)
print "Std",np.mean(finalStd)
print "PCfs",np.mean(finalPcfs)
print "Time", np.mean(finalTime)
"""
# *********************
"""
   #Polyhedral Conic Functions Example
# PCF algorithm requires both data clusters
# This function seperates data to clusters according to their labels
def seperatetoAB(data, labels, indexes):
    A = []
    B = []
    for i in indexes:
        if labels[i] == -1:
            A.append(data[i])
        elif labels[i] == 1:
            B.append(data[i])
    A = np.array(A)
    B = np.array(B)
    return A, B


# This function converts given labels (l1,l2) to 1, -1
def convertLabels(labelData, l1, l2):
    lbls = np.zeros(len(labelData))
    for i in range(len(labelData)):
        if labelData[i] == l1:
            lbls[i] = -1
        elif labelData[i] == l2:
            lbls[i] = 1
    return lbls
finalTest =[]
finalTRain=[]
finalStd=[]
finalTime=[]
finalPcfs=[]
kats= np.linspace(1,1,10)
for kt in kats:
    f = open('Heart.txt')
    X = []
    labels = []
    for line in f:
        row = []
        line = line.split(' ')
        labels.append(int(line[-1]))
        X.append([float(line[i]) for i in range(len(line)-1)])
    labels = np.array(labels)
    X = np.array(X)

    acc = []
    timeS = []
    timeF = []
    pcfNumber=[]
    labels = convertLabels(labels, 1, 2)
    skf = StratifiedKFold(n_splits=10)
    accTrain = []
    for train, test in skf.split(X,labels):
        sepData = seperatetoAB(X, labels, train)
        timeS.append(time.time())
        pModel = pc.PCF_iterative()
        pModel.fit(sepData[0], sepData[1])
        timeF.append(time.time())
        pcfNumber.append(len( pModel.pcfs))
        acc.append(accuracy_score(labels[test], pModel.predict(X[test],-1,1)))
        accTrain.append(accuracy_score(labels[train],pModel.predict(X[train],-1,1)))
    print "Training Acc:", sum(accTrain)/len(accTrain)
    print "Test Acc:", sum(acc)/len(acc)
    print "Std Acc", np.std(acc)
    print "Training Time",sum([timeF[i]-timeS[i] for i in range(len(timeS))])/len(timeS)
    print "**************************************************kt: ",kt
    print "Training Acc: ", sum(accTrain)/len(accTrain)
    finalTRain.append(np.mean(accTrain))
    print '\033[1m' + 'Test Accuracy'+str((sum(acc)/ len(acc))) + '\033[0m'
    finalTest.append(np.mean(acc))
    print "std Acc: ", np.std(acc)
    finalStd.append(np.std(acc))
    print "Training Time",sum([timeF[i]-timeS[i] for i in range(len(timeS))])/len(timeS)
    finalTime.append(sum([timeF[i]-timeS[i] for i in range(len(timeS))])/len(timeS))
    print "Av. Number PCF",np.mean(pcfNumber)
    finalPcfs.append(np.mean(pcfNumber))
print "---------------------Final---------------kt: ",kt
print "Training", np.mean(finalTRain)
print "Test",np.mean(finalTest)
print "Std",np.mean(finalStd)
print "PCfs",np.mean(finalPcfs)
print "Time", np.mean(finalTime)
"""
    #***********************
"""
    #Polyhedral Conic Functions Example
# PCF algorithm requires both data clusters
# This function seperates data to clusters according to their labels
def seperatetoAB(data, labels, indexes):
    A = []
    B = []
    for i in indexes:
        if labels[i] == -1:
            A.append(data[i])
        elif labels[i] == 1:
            B.append(data[i])
    A = np.array(A)
    B = np.array(B)
    return A, B


# This function converts given labels (l1,l2) to 1, -1
def convertLabels(labelData, l1, l2):
    lbls = np.zeros(len(labelData))
    for i in range(len(labelData)):
        if labelData[i] == l1:
            lbls[i] = -1
        elif labelData[i] == l2:
            lbls[i] = 1
    return lbls
finalTest =[]
finalTRain=[]
finalStd=[]
finalTime=[]
finalPcfs=[]
kats= np.linspace(1,1,10)
for kt in kats:
    f = open('Ionosphere.txt')
    X = []
    labels = []
    for line in f:
        row = []
        line = line.split(',')
        labels.append(line[-1].replace('\n', ''))
        X.append([float(line[i]) for i in range(len(line)-1)])
    labels = np.array(labels)
    X = np.array(X)
    pcfNumber=[]
    acc = []
    timeS = []
    timeF = []
    counts = []
    labels = convertLabels(labels, 'g', 'b')
    skf = StratifiedKFold(n_splits=10)
    accTrain = []
    for train, test in skf.split(X,labels):
        sepData = seperatetoAB(X, labels, train)
        timeS.append(time.time())
        pModel = pc.PCF_iterative()
        pModel.fit(sepData[0], sepData[1])
        timeF.append(time.time())
        pcfNumber.append(len( pModel.pcfs))
        acc.append(accuracy_score(labels[test], pModel.predict(X[test],-1,1)))
        accTrain.append(accuracy_score(labels[train],pModel.predict(X[train],-1,1)))
    print "Training Acc:", sum(accTrain)/len(accTrain)
    print "Test Acc:", sum(acc)/len(acc)
    print "Std Acc", np.std(acc)
    print "Training Time",sum([timeF[i]-timeS[i] for i in range(len(timeS))])/len(timeS)
    print "**************************************************kt: ",kt
    print "Training Acc: ", sum(accTrain)/len(accTrain)
    finalTRain.append(np.mean(accTrain))
    print '\033[1m' + 'Test Accuracy'+str((sum(acc)/ len(acc))) + '\033[0m'
    finalTest.append(np.mean(acc))
    print "std Acc: ", np.std(acc)
    finalStd.append(np.std(acc))
    print "Training Time",sum([timeF[i]-timeS[i] for i in range(len(timeS))])/len(timeS)
    finalTime.append(sum([timeF[i]-timeS[i] for i in range(len(timeS))])/len(timeS))
    print "Av. Number PCF",np.mean(pcfNumber)
    finalPcfs.append(np.mean(pcfNumber))
print "---------------------Final---------------kt: ",kt
print "Training", np.mean(finalTRain)
print "Test",np.mean(finalTest)
print "Std",np.mean(finalStd)
print "PCfs",np.mean(finalPcfs)
print "Time", np.mean(finalTime)
"""
# *********************
"""
    #Polyhedral Conic Functions Example
# PCF algorithm requires both data clusters
# This function seperates data to clusters according to their labels
def seperatetoAB(data, labels, indexes):
    A = []
    B = []
    for i in indexes:
        if labels[i] == -1:
            A.append(data[i])
        elif labels[i] == 1:
            B.append(data[i])
    A = np.array(A)
    B = np.array(B)
    return A, B


# This function converts given labels (l1,l2) to 1, -1
def convertLabels(labelData, l1, l2):
    lbls = np.zeros(len(labelData))
    for i in range(len(labelData)):
        if labelData[i] == l1:
            lbls[i] = -1
        elif labelData[i] == l2:
            lbls[i] = 1
    return lbls
finalTest =[]
finalTRain=[]
finalStd=[]
finalTime=[]
finalPcfs=[]
kats= np.linspace(1,1,10)
for kt in kats:
    f = open('WBCP32Features.txt')
    X = []
    labels = []
    for line in f:
        row = []
        line = line.split(',')
        labels.append(line[1])
        X.append([float(line[i]) for i in range(2,len(line))])
    labels = np.array(labels)
    X = np.array(X)

    acc = []
    timeS = []
    timeF = []
    counts = []
    labels = convertLabels(labels, 'N', 'R')
    skf = StratifiedKFold(n_splits=10)
    accTrain = []
    pcfNumber=[]
    for train, test in skf.split(X,labels):
        sepData = seperatetoAB(X, labels, train)
        timeS.append(time.time())
        pModel = pc.PCF_iterative()
        pModel.fit(sepData[0], sepData[1])
        timeF.append(time.time())
        pcfNumber.append(len( pModel.pcfs))
        acc.append(accuracy_score(labels[test], pModel.predict(X[test],-1,1)))
        accTrain.append(accuracy_score(labels[train],pModel.predict(X[train],-1,1)))
    print "Training Acc:", sum(accTrain)/len(accTrain)
    print "Test Acc:", sum(acc)/len(acc)
    print "Std Acc", np.std(acc)
    print "Training Time",sum([timeF[i]-timeS[i] for i in range(len(timeS))])/len(timeS)
    print "**************************************************kt: ",kt
    print "Training Acc: ", sum(accTrain)/len(accTrain)
    finalTRain.append(np.mean(accTrain))
    print '\033[1m' + 'Test Accuracy'+str((sum(acc)/ len(acc))) + '\033[0m'
    finalTest.append(np.mean(acc))
    print "std Acc: ", np.std(acc)
    finalStd.append(np.std(acc))
    print "Training Time",sum([timeF[i]-timeS[i] for i in range(len(timeS))])/len(timeS)
    finalTime.append(sum([timeF[i]-timeS[i] for i in range(len(timeS))])/len(timeS))
    print "Av. Number PCF",np.mean(pcfNumber)
    finalPcfs.append(np.mean(pcfNumber))
print "---------------------Final---------------kt: ",kt
print "Training", np.mean(finalTRain)
print "Test",np.mean(finalTest)
print "Std",np.mean(finalStd)
print "PCfs",np.mean(finalPcfs)
print "Time", np.mean(finalTime)
"""
# ****"*****************

  #Polyhedral Conic Functions Example
#PCF algorithm requires both data clusters
#This function seperates data to clusters according to their labels
def seperatetoAB(data, labels, indexes):
    A = []
    B = []
    for i in indexes:
        if labels[i] == -1:
            A.append(data[i])
        elif labels[i] == 1:
            B.append(data[i])
    A = np.array(A)
    B = np.array(B)
    return A, B

#This function converts given labels (l1,l2) to 1, -1
def convertLabels(labelData, l1, l2):
    lbls = np.zeros(len(labelData))
    for i in range(len(labelData)):
        if labelData[i] == l1:
            lbls[i] = -1
        elif labelData[i] == l2:
            lbls[i] = 1
    return lbls
finalTest =[]
finalTRain=[]
finalStd=[]
finalTime=[]
finalPcfs=[]
kats= np.linspace(1,1,10)
for kt in kats:
    f = open('WBCD9Features.txt')
    X = []
    labels = []
    for line in f:
        row = []
        line = line.split(',')
        labels.append(int(line[-1]))
        X.append([float(line[i]) for i in range(1,len(line)-1)])
    labels = np.array(labels)
    X = np.array(X)

    acc = []
    timeS = []
    timeF = []
    counts = []
    labels = convertLabels(labels, 4, 2)
    skf = StratifiedKFold(n_splits=10)
    accTrain = []
    pcfNumber=[]
    for train, test in skf.split(X,labels):
        sepData = seperatetoAB(X, labels, train)
        timeS.append(time.time())
        pModel = pc.PCF_iterative()
        pModel.fit(sepData[0], sepData[1])
        timeF.append(time.time())
        pcfNumber.append(len( pModel.pcfs))
        acc.append(accuracy_score(labels[test], pModel.predict(X[test],-1,1)))
        accTrain.append(accuracy_score(labels[train],pModel.predict(X[train],-1,1)))
    print "Training Acc:", sum(accTrain)/len(accTrain)
    print "Test Acc:", sum(acc)/len(acc)
    print "Std Acc", np.std(acc)
    print "Training Time",sum([timeF[i]-timeS[i] for i in range(len(timeS))])/len(timeS)
    print "**************************************************kt: ",kt
    print "Training Acc: ", sum(accTrain)/len(accTrain)
    finalTRain.append(np.mean(accTrain))
    print '\033[1m' + 'Test Accuracy'+str((sum(acc)/ len(acc))) + '\033[0m'
    finalTest.append(np.mean(acc))
    print "std Acc: ", np.std(acc)
    finalStd.append(np.std(acc))
    print "Training Time",sum([timeF[i]-timeS[i] for i in range(len(timeS))])/len(timeS)
    finalTime.append(sum([timeF[i]-timeS[i] for i in range(len(timeS))])/len(timeS))
    print "Av. Number PCF",np.mean(pcfNumber)
    finalPcfs.append(np.mean(pcfNumber))
print "---------------------Final---------------kt: ",kt
print "Training", np.mean(finalTRain)
print "Test",np.mean(finalTest)
print "Std",np.mean(finalStd)
print "PCfs",np.mean(finalPcfs)
print "Time", np.mean(finalTime)
"""
# *********************

    #Polyhedral Conic Functions Example

#PCF algorithm requires both data clusters
#This function seperates data to clusters according to their labels
def seperatetoAB(data, labels, indexes):
    A = []
    B = []
    for i in indexes:
        if labels[i] == -1:
            A.append(data[i])
        elif labels[i] == 1:
            B.append(data[i])
    A = np.array(A)
    B = np.array(B)
    return A, B

#This function converts given labels (l1,l2) to 1, -1
def convertLabels(labelData, l1, l2):
    lbls = np.zeros(len(labelData))
    for i in range(len(labelData)):
        if labelData[i] == l1:
            lbls[i] = -1
        elif labelData[i] == l2:
            lbls[i] = 1
    return lbls
finalTest =[]
finalTRain=[]
finalStd=[]
finalTime=[]
finalPcfs=[]
kats= np.linspace(1,1,10)
for kt in kats:
    f = open('BupaLiver.txt')
    X = []
    labels = []
    for line in f:
        row = []
        line = line.split(',')
        labels.append(int(line[-1]))
        X.append([float(line[i]) for i in range(len(line)-1)])
    labels = np.array(labels)
    X = np.array(X)

    acc = []
    timeS = []
    timeF = []
    counts = []
    labels = convertLabels(labels, 2, 1)
    skf = StratifiedKFold(n_splits=10)
    accTrain = []
    pcfNumber=[]
    for train, test in skf.split(X,labels):
        sepData = seperatetoAB(X, labels, train)
        timeS.append(time.time())
        pModel = pc.PCF_iterative()
        pModel.fit(sepData[0], sepData[1])
        timeF.append(time.time())
        pcfNumber.append(len( pModel.pcfs))
        acc.append(accuracy_score(labels[test], pModel.predict(X[test],-1,1)))
        accTrain.append(accuracy_score(labels[train],pModel.predict(X[train],-1,1)))
    print "Training Acc:", sum(accTrain)/len(accTrain)
    print "Test Acc:", sum(acc)/len(acc)
    print "Std Acc", np.std(acc)
    print "Training Time",sum([timeF[i]-timeS[i] for i in range(len(timeS))])/len(timeS)
    print "**************************************************kt: ",kt
    print "Training Acc: ", sum(accTrain)/len(accTrain)
    finalTRain.append(np.mean(accTrain))
    print '\033[1m' + 'Test Accuracy'+str((sum(acc)/ len(acc))) + '\033[0m'
    finalTest.append(np.mean(acc))
    print "std Acc: ", np.std(acc)
    finalStd.append(np.std(acc))
    print "Training Time",sum([timeF[i]-timeS[i] for i in range(len(timeS))])/len(timeS)
    finalTime.append(sum([timeF[i]-timeS[i] for i in range(len(timeS))])/len(timeS))
    print "Av. Number PCF",np.mean(pcfNumber)
    finalPcfs.append(np.mean(pcfNumber))
print "---------------------Final---------------kt: ",kt
print "Training", np.mean(finalTRain)
print "Test",np.mean(finalTest)
print "Std",np.mean(finalStd)
print "PCfs",np.mean(finalPcfs)
print "Time", np.mean(finalTime)
"""
# *****************************************************************************************************************
