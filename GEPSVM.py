import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import scipy

#A,B: training data sets, center: center of the closest dataset A/B, t: Thikonov term
class GEPSVM:

    def fit(self, A, B ,center,t):
        self.center = center

        self.A = A
        self.B = B

        self._preProcess()
        G = self.A.T.dot(self.A)

        tikhonov = t
        I = np.identity(len(G))
        G = G + I*tikhonov

        H = self.B.T.dot(self.B)


        eigvals, eigvecs = scipy.linalg.eig(G, H)
        eigvals = eigvals.real
        eigvecs =  eigvecs.real
        mineigenindex = np.argmin(eigvals)

        self.w = eigvecs[:,mineigenindex][:len(A[0])]
        self.ksi = eigvecs[:,mineigenindex][len(A[0]):-1]
        self.gamma = eigvecs[:,mineigenindex][-1]



        return self



    def _preProcess(self):

        allKsiA=np.absolute(np.subtract(self.A, self.center))
        self.A = np.append(self.A, allKsiA, axis=1)
        self.A = np.append(self.A, -1 * np.ones((len(self.A), 1)), axis=1)

        allKsiB=np.absolute(np.subtract(self.B, self.center))
        self.B = np.append(self.B, allKsiB, axis=1)
        self.B = np.append(self.B, -1 * np.ones((len(self.B), 1)), axis=1)




#p1: model 1, p2: model 2, X: prediction data, lb1: label 1, lb2: label 2
def predict(p1,p2,X,lb1,lb2):
    result = np.ones(len(X))
    for i in range(len(X)):

        result1 = np.dot(np.subtract(X[i], p1.center), p1.w) + np.dot(p1.ksi, np.absolute(np.subtract(X[i], p1.center))) - p1.gamma
        result2 = np.dot(np.subtract(X[i], p2.center), p2.w) + np.dot(p2.ksi, np.absolute(np.subtract(X[i], p2.center))) - p2.gamma


        if abs(result1) < abs(result2):
            result[i] = lb1
        else:
            result[i] = lb2
    return result