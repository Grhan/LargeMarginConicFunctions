import numpy as np

w = np.ones(3)
w[1] =3
w[0] =2
w[2] = 5


"""
from sklearn import svm
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D


np.seterr(divide='ignore', invalid='ignore')

#A = [[0,0],[1,1]]
#B= [[1,0], [0,1]]

A= [[2,0],[0,2],[-2,0],[0,-2],[5,0],[0,5],[-5,0],[0,-5],[0,0]]
B =[[10,0],[0,10],[-10,0],[0,-10],[15,0],[0,15],[-15,0],[0,-15]]
center =[0,0]
A = np.array(A)
B = np.array(B)


l1Column = np.linalg.norm(np.subtract(A, center), ord=1, axis=1, keepdims=True)
A = np.append(A, l1Column, axis=1)
l1Column = np.linalg.norm(np.subtract(B, center), ord=1, axis=1, keepdims=True)
B = np.append(B, l1Column, axis=1)
A = np.append(A, -1 * np.ones((len(A), 1)), axis=1)
B = np.append(B, -1 * np.ones((len(B), 1)), axis=1)


G = A.T.dot(A)

tikhonov = 0.2
I = np.identity(len(G))
G = G + I * tikhonov

H = B.T.dot(B)



eigvals, eigvecs = scipy.linalg.eig(G, H)
eigvals = eigvals.real
eigvecs =  eigvecs.real
mineigenindex = np.argmin(eigvals)
print  mineigenindex
print eigvals.astype(float)
print eigvecs.astype(float)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(A[:,0], A[:,1],  [0], c='r', marker='o')
ax.scatter(B[:,0], B[:,1], [0], c='b', marker='^')


normal = eigvecs[:,mineigenindex][:2]

x = y = np.arange(-15, 15.0,1)
X, Y = np.meshgrid(x, y)
w = normal

#Z=  w[0]*X + w[1]*Y -eigvecs[:,mineigenindex][-1]
Z= w[0]*X + w[1]*Y+  eigvecs[:,mineigenindex][2]*(abs(X-center[0])+abs(Y-center[1]))-eigvecs[:,mineigenindex][3]
ax.plot_surface(X, Y, Z, rstride=5, cstride=5, alpha=0.2, linewidth=0.8, )



plt.show()"""