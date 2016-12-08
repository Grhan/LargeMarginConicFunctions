import numpy as np
import PSVM
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
import time
import GEPSVM
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import  math

def pValue(p1,p2,X):
    result = np.zeros(len(X))

    for i in range(len(X)):
        signs = np.sign(np.subtract(X[i], p1.center))
        g1 = p1.gamma- p1.w.T.dot(p1.center) - signs.T.dot(p1.ksi*p1.center)
        w1 = p1.w + signs * p1.ksi
        signs = np.sign(np.subtract(X[i], p2.center))
        g2 = p2.gamma - p2.w.T.dot(p2.center) - signs.T.dot(p2.ksi * p2.center)
        w2 = p1.w + signs * p2.ksi

        v1 = abs(np.subtract(X[i], p1.center).T.dot(w1) -g1) / np.linalg.norm(w1, ord=2)
        v2 = abs(np.subtract(X[i], p2.center).T.dot(w2) - g2) / np.linalg.norm(w2, ord=2)
        if v2<v1:
            result[i] = 1
    return result

A= [[0,2],[2,0],[5,0],[0,5],[-2,0],[-5,0],[0,-2],[0,-5],[0,0]]
B =[[10,0],[0,10],[-10,0],[0,-10],[15,0],[0,15],[-15,0],[0,-15]]



A = np.array(A)
B = np.array(B)
center1 =np.zeros(2)
center1[0] = sum(A[:,0])/len(A)
center1[1] =sum(A[:,1])/len(A)
center2 =np.zeros(2)
center2[0] =sum(B[:,0])/len(B)
center2[1] =sum(B[:,1])/len(B)


pModel = GEPSVM.GEPSVM()
pModel.fit(A,B,center1)
pModel1 = GEPSVM.GEPSVM()
pModel1.fit(B,A,center2)


print "Model 1 W:",pModel.w," ksi:",pModel.ksi,"  gamma:",pModel.gamma , "  center:", center1
print "Model 2 W:",pModel1.w," ksi:",pModel1.ksi,"  gamma:",pModel1.gamma, "    center", center2

print "A model1 vs model2 ", pValue(pModel,pModel1,A)

print "B model2 vs model1 ", pValue(pModel,pModel1,B)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(A[:,0], A[:,1],  [0], c='r', marker='o')
ax.scatter(B[:,0], B[:,1], [0], c='b', marker='^')




x  =y= np.arange(-20, 25.0,1)

X, Y = np.meshgrid(x, y)
w = pModel.w
Z=  w[0]*(X-center1[0])+ w[1]*(Y -center1[1])+ pModel.ksi*(abs(X-center1[0])+abs(Y-center1[1]))- pModel.gamma
ax.plot_surface(X, Y, Z, rstride=5, cstride=5, alpha=0.2, linewidth=0.8, )


w = pModel1.w
Z=  w[0]*(X-center2[0])+ w[1]*(Y -center2[1])+ pModel1.ksi*(abs(X-center2[0])+abs(Y-center2[1]))-pModel1.gamma

ax.plot_surface(X, Y, Z, rstride=5, cstride=5, alpha=0.2, linewidth=0.8, )



plt.show()
""""
f = open('pima-indians-diabetes.data.txt')
X = []
labels = []
A = []
B = []


for line in f:

    line = line.split(',')
    labels.append(int(line[-1]))

    X.append([float(line[i]) for i in range(len(line)-1)])
    if labels[-1] == 1:

        A.append(X[-1])
    else:

        B.append(X[-1])

labels = np.array(labels)
X = np.array(X)
A = np.array(A)
B = np.array(B)




acc = []
timeS = []
timeF = []
models = []


pModel = GEPSVM.GEPSVM()
pModel.fit(A,B)
pModel1 = GEPSVM.GEPSVM()
pModel1.fit(B,A)

predictions = GEPSVM.predict(pModel,pModel1,X )


print accuracy_score(labels, predictions)
"""

