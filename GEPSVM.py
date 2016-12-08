import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import scipy

class GEPSVM:

    def fit(self, A, B ,center):
        self.center = center

        self.A = A
        self.B = B

        self._preProcess()

        G = self.A.T.dot(self.A)

        tikhonov = 0.8
        I = np.identity(len(G))
        G = G + I*tikhonov

        H = self.B.T.dot(self.B)


        eigvals, eigvecs = scipy.linalg.eig(G, H)
        eigvals = eigvals.real
        eigvecs =  eigvecs.real
        mineigenindex = np.argmin(eigvals)

        self.gamma = eigvecs[:,mineigenindex][3]
        self.ksi = eigvecs[:,mineigenindex][2]
        self.w = (eigvecs[:,mineigenindex])[:2]

        return self



    def _preProcess(self):

        l1Column = np.linalg.norm(np.subtract(self.A, self.center), ord=1 ,axis=1 ,keepdims=True)
        self.A = np.append(self.A, l1Column, axis=1)
        self.A = np.append(self.A, -1 * np.ones((len(self.A), 1)), axis=1)

        l1ColumnB = np.linalg.norm(np.subtract(self.B, self.center), ord=1, axis=1, keepdims=True)
        self.B = np.append(self.B, l1ColumnB, axis=1)
        self.B = np.append(self.B, -1 * np.ones((len(self.B), 1)), axis=1)

def predict(p1,p2, X):
    result = np.ones(len(X))
    for i in range(len(X)):
        sub = np.subtract(X[i], p1.center)
        signs = np.sign(sub)
        v1 = X[i].T.dot(p1.w)+ p1.ksi() - p1.gamma
        v2 = X[i].T.dot(p2.w) - p2.gamma
        if abs(v2) < abs(v1):
            result[i] = 0
    return result