import numpy as np
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import time
from sklearn.metrics import precision_recall_fscore_support

def predict(X,center,w,ksi,gamma, labelA, labelB):
    predictions = list()
    for i in range(len(X)):

        f = np.dot(np.subtract(X[i], center), w) + abs(ksi) * np.linalg.norm(np.subtract(X[i],center),
                                                                              ord=1) - abs( gamma)
        if f <= 0.0:
            f = labelA
        else:
            f = labelB
        predictions.append(f)
    return predictions

def calcCenter(dt,ind,lab,thlab):
    center =np.zeros((len(dt[0])))
    counter =0
    for i in ind:
        if lab[i]==thlab:
            center += dt[i]
            counter += 1
    return center/counter
"""
print "***************************** PimaDiabetes"
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

for param in range(14):
    st = (10**(param+1))*1.e-08
    end =(10**(param+1))*1.e-07
    print "****************************st,end:",st,end
    kats= np.linspace( st ,end,num=10)
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
        counts = []
        skf = StratifiedKFold(n_splits=10)
        accTrain = []

        for train, test in skf.split(X,labels):
            timeS.append(time.time())
            #centerTrain = np.mean(X[train], axis=0)
            centerTrain = calcCenter(X, train, labels, 1)
            l1Train = np.linalg.norm(np.subtract(X[train], centerTrain), ord=1, axis=1, keepdims=True)
            tempX = np.append(X[train], l1Train, axis=1)
            svc = svm.LinearSVC(C=kt).fit(tempX, labels[train])
            timeF.append(time.time())
            w=svc.coef_[0]
            acc.append(accuracy_score(labels[test], predict(X[test],centerTrain,w[:-1],w[-1],svc.intercept_[0],0,1)))
            accTrain.append(accuracy_score(labels[train], predict(X[train], centerTrain, w[:-1], w[-1], svc.intercept_[0], 0, 1)))

        print "**************************************************kt: ",kt
        print "Training Acc: ", sum(accTrain)/len(accTrain)
        print '\033[1m' + 'Test Accuracy'+str((sum(acc)/ len(acc))) + '\033[0m'
        print "std Acc: ", np.std(acc)
        print "Training Time",sum([timeF[i]-timeS[i] for i in range(len(timeS))])/len(timeS)
"""
"""
print "***************************** Heart"
# *********************

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
for param in range(14):
    st = (10**(param+1))*1.e-08
    end =(10**(param+1))*1.e-07
    print "****************************st,end:",st,end
    kats= np.linspace( st ,end,num=10*(param+1))
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
        counts = []
        skf = StratifiedKFold(n_splits=10)
        accTrain = []

        for train, test in skf.split(X,labels):
            timeS.append(time.time())
            centerTrain = np.mean(X[train], axis=0)
            #centerTrain = calcCenter(X, train, labels, 1)
            l1Train = np.linalg.norm(np.subtract(X[train], centerTrain), ord=1, axis=1, keepdims=True)
            tempX = np.append(X[train], l1Train, axis=1)
            svc = svm.LinearSVC(C=kt).fit(tempX, labels[train])
            timeF.append(time.time())
            w=svc.coef_[0]
            acc.append(accuracy_score(labels[test], predict(X[test],centerTrain,w[:-1],w[-1],svc.intercept_[0],1,2)))
            accTrain.append(accuracy_score(labels[train], predict(X[train], centerTrain, w[:-1], w[-1], svc.intercept_[0], 1 ,2)))

        print "**************************************************kt: ",kt
        print "Training Acc: ", sum(accTrain)/len(accTrain)
        print '\033[1m' + 'Test Accuracy'+str((sum(acc)/ len(acc))) + '\033[0m'
        print "std Acc: ", np.std(acc)
        print "Training Time",sum([timeF[i]-timeS[i] for i in range(len(timeS))])/len(timeS)
"""

    #***********************
"""
print "***************************** Ionosphere"
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


for param in range(14):
    st = (10**(param+1))*1.e-08
    end =(10**(param+1))*1.e-07
    print "****************************st,end:",st,end
    kats= np.linspace(  st , end,num=10*(param+1))
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
        labels = convertLabels(labels, 'g', 'b')
        acc = []
        timeS = []
        timeF = []
        counts = []
        skf = StratifiedKFold(n_splits=10)
        accTrain = []

        for train, test in skf.split(X,labels):
            timeS.append(time.time())
            #centerTrain = np.mean(X[train], axis=0)
            centerTrain = calcCenter(X, train, labels, 1)
            l1Train = np.linalg.norm(np.subtract(X[train], centerTrain), ord=1, axis=1, keepdims=True)
            tempX = np.append(X[train], l1Train, axis=1)
            svc = svm.LinearSVC(C=kt).fit(tempX, labels[train])
            timeF.append(time.time())
            w=svc.coef_[0]
            acc.append(accuracy_score(labels[test], predict(X[test],centerTrain,w[:-1],w[-1],svc.intercept_[0],-1,1)))
            accTrain.append(accuracy_score(labels[train], predict(X[train], centerTrain, w[:-1], w[-1], svc.intercept_[0], -1, 1)))

        print "**************************************************kt: ",kt
        print "Training Acc: ", sum(accTrain)/len(accTrain)
        print '\033[1m' + 'Test Accuracy'+str((sum(acc)/ len(acc))) + '\033[0m'
        print "std Acc: ", np.std(acc)
        print "Training Time",sum([timeF[i]-timeS[i] for i in range(len(timeS))])/len(timeS)



"""

"""
# *********************
print "***************************** WBCP32Features"
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

for param in range(14):
    st = (10**(param+1))*1.e-08
    end =(10**(param+1))*1.e-07
    print "****************************st,end:",st,end
    kats= np.linspace(  st , end,num=10*(param+1))
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
        labels = convertLabels(labels, 'R', 'N')
        acc = []
        timeS = []
        timeF = []
        counts = []
        skf = StratifiedKFold(n_splits=10)
        accTrain = []

        for train, test in skf.split(X,labels):
            timeS.append(time.time())
            #centerTrain = np.mean(X[train], axis=0)
            centerTrain = calcCenter(X, train, labels, -1)
            l1Train = np.linalg.norm(np.subtract(X[train], centerTrain), ord=1, axis=1, keepdims=True)
            tempX = np.append(X[train], l1Train, axis=1)
            svc = svm.LinearSVC(C=kt).fit(tempX, labels[train])
            timeF.append(time.time())
            w=svc.coef_[0]
            acc.append(accuracy_score(labels[test], predict(X[test],centerTrain,w[:-1],w[-1],svc.intercept_[0],-1,1)))
            accTrain.append(accuracy_score(labels[train], predict(X[train], centerTrain, w[:-1], w[-1], svc.intercept_[0], -1, 1)))

        print "**************************************************kt: ",kt
        print "Training Acc: ", sum(accTrain)/len(accTrain)
        print '\033[1m' + 'Test Accuracy'+str((sum(acc)/ len(acc))) + '\033[0m'
        print "std Acc: ", np.std(acc)
        print "Training Time",sum([timeF[i]-timeS[i] for i in range(len(timeS))])/len(timeS)

"""
"""
# *********************
print "***************************** WBCD9Features"
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

for param in range(14):
    st = (10**(param+1))*1.e-08
    end =(10**(param+1))*1.e-07
    print "****************************st,end:",st,end
    kats= np.linspace(  st , end,num=10*(param+1))
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
        skf = StratifiedKFold(n_splits=10)
        accTrain = []

        for train, test in skf.split(X,labels):
            timeS.append(time.time())
            #centerTrain = np.mean(X[train], axis=0)
            centerTrain = calcCenter(X, train, labels, 2)
            l1Train = np.linalg.norm(np.subtract(X[train], centerTrain), ord=1, axis=1, keepdims=True)
            tempX = np.append(X[train], l1Train, axis=1)
            svc = svm.LinearSVC(C=kt).fit(tempX, labels[train])
            timeF.append(time.time())
            w=svc.coef_[0]
            acc.append(accuracy_score(labels[test], predict(X[test],centerTrain,w[:-1],w[-1],svc.intercept_[0],2,4)))
            accTrain.append(accuracy_score(labels[train], predict(X[train], centerTrain, w[:-1], w[-1], svc.intercept_[0], 2, 4)))

        print "**************************************************kt: ",kt
        print "Training Acc: ", sum(accTrain)/len(accTrain)
        print '\033[1m' + 'Test Accuracy'+str((sum(acc)/ len(acc))) + '\033[0m'
        print "std Acc: ", np.std(acc)
        print "Training Time",sum([timeF[i]-timeS[i] for i in range(len(timeS))])/len(timeS)

# *********************
"""
"""
print "***************************** BupaLiver"
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

for param in range(14):
    st = (10**(param+1))*1.e-08
    end =(10**(param+1))*1.e-07
    print "****************************st,end:",st,end
    kats= np.linspace(  st , end,num=10*(param+1))
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
        skf = StratifiedKFold(n_splits=10)
        accTrain = []

        for train, test in skf.split(X,labels):
            timeS.append(time.time())
            #centerTrain = np.mean(X[train], axis=0)
            centerTrain = calcCenter(X, train, labels, 1)
            l1Train = np.linalg.norm(np.subtract(X[train], centerTrain), ord=1, axis=1, keepdims=True)
            tempX = np.append(X[train], l1Train, axis=1)
            svc = svm.LinearSVC(C=kt).fit(tempX, labels[train])
            timeF.append(time.time())
            w=svc.coef_[0]
            acc.append(accuracy_score(labels[test], predict(X[test],centerTrain,w[:-1],w[-1],svc.intercept_[0],1,2)))
            accTrain.append(accuracy_score(labels[train], predict(X[train], centerTrain, w[:-1], w[-1], svc.intercept_[0], 1, 2)))

        print "**************************************************kt: ",kt
        print "Training Acc: ", sum(accTrain)/len(accTrain)
        print '\033[1m' + 'Test Accuracy'+str((sum(acc)/ len(acc))) + '\033[0m'
        print "std Acc: ", np.std(acc)
        print "Training Time",sum([timeF[i]-timeS[i] for i in range(len(timeS))])/len(timeS)

# *****************************************************************************************************************
"""



