import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


import GEPSVM as pc


import time


    # Polyhedral Conic Functions Example
# PCF algorithm requires both data clusters
# This function seperates data to clusters according to their labels
def seperatetoAB(data, labels, indexes):
    A = []
    B = []
    for i in indexes:
        if labels[i] == 0:
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
            lbls[i] = 0
        elif labelData[i] == l2:
            lbls[i] = 1
    return lbls
"""



    f = open('Ionosphere.txt')
    X = []
    labels = []
    for line in f:
        row = []
        line = line.split(',')
        labels.append(line[-1].replace('\n', ''))
        X.append([float(line[i]) for i in range(len(line)-1)])

    f = open('PimaDiabetes.txt')
    X = []
    labels = []
    for line in f:
        row = []
        line = line.split(',')
        labels.append(int(line[-1]))
        X.append([float(line[i]) for i in range(len(line)-1)])
    f = open('Heart.txt')
    X = []
    labels = []
    for line in f:
        row = []
        line = line.split(' ')
        labels.append(int(line[-1]))
        X.append([float(line[i]) for i in range(len(line)-1)])
    f = open('WBCP32Features.txt')
    X = []
    labels = []
    for line in f:
        row = []
        line = line.split(',')
        labels.append(line[1])
        X.append([float(line[i]) for i in range(2,len(line))])
    f = open('WBCD9Features.txt')
    X = []
    labels = []
    for line in f:
        row = []
        line = line.split(',')
        labels.append(int(line[-1]))
        X.append([float(line[i]) for i in range(1,len(line)-1)])
    f = open('BupaLiver.txt')
    X = []
    labels = []
    for line in f:
        row = []
        line = line.split(',')
        labels.append(int(line[-1]))
        X.append([float(line[i]) for i in range(len(line)-1)])
"""


for param in range(14):
    st = (10**(param+1))*1.e-08
    end =(10**(param+1))*1.e-07
    print "****************************st,end:",st,end
    tiks= np.linspace( st ,end,num=10)

    for tk in tiks:
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
        accTrain = []
        timeS = []
        timeF = []
        labels = convertLabels(labels, 2, 1)
        skf = StratifiedKFold(n_splits=10)
        for train, test in skf.split(X,labels):
            sepData = seperatetoAB(X, labels, train)
            timeS.append(time.time())
            #centerTrain= np.mean(X[train], axis=0)
            center1 = np.mean(sepData[0], axis=0)
            pModel1 = pc.GEPSVM()
            pModel1.fit(sepData[0], sepData[1],center1,tk)
            center2=np.mean(sepData[1], axis=0)
            pModel2 = pc.GEPSVM()
            pModel2.fit(sepData[1], sepData[0],center2,tk)
            timeF.append(time.time())
            acc.append(accuracy_score(labels[test], pc.predict(pModel1,pModel2,X[test],0,1)))
            accTrain.append(accuracy_score(labels[train],pc.predict(pModel1,pModel2,X[train],0,1)))
        print "**************************tk:",tk
        print "Time", (sum(timeF[i]-timeS[i] for i in range(len(timeF)))/len(timeF))
        print "Training Acc", (sum(accTrain)/ len(accTrain))
        print accTrain
        print  '\033[1m' + 'Test Accuracy'+str((sum(acc)/ len(acc))) + '\033[0m'
        print  acc
        print np.std(acc)

