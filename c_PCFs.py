import numpy as np
import math
import random
from gurobipy import *

"""
    -Seperation via polyhedral conic functions, Gasimov and Ozturk, 2006
    -Implementation includes two classes PCF and PCF_iterative.
    -PCF solves part of the defined problem and returns only one conic function for given A,B datasets and a random center.
    -PCF_iterative  solves whole problem in a iterative way as described in the paper and returns sets of conic functions.
    -To execute this algortihm Gurobi solver and gurobi.py are required
     http://www.gurobi.com/
     https://www.gurobi.com/documentation/6.5/quickstart_mac/the_gurobi_python_interfac.html
"""


class PCF:
    def __init__(self):
        # type: () -> object
        self.w = list()
        self.gamma = 0
        #self.ksi = list()
        self.ksi=0

    def setParam(self, A, B,center,c,cb):
        # set parameters dimension = number of features
        self.center = center
        dimension = len(A[0])
        # m =  s(A), p = s(B)
        m = len(A)
        p = len(B)

        #create gurobi model
        model = Model()
        model.setParam("OutputFlag",0)
        #add gamma and ksi variables
        gamma = model.addVar(vtype=GRB.CONTINUOUS, lb=1, name='gamma')
        ksi = model.addVar(vtype=GRB.CONTINUOUS, lb=0,name='ksi')
        #add w variables
        w = list()
        for i in range(dimension):
            w.append(model.addVar(vtype=GRB.CONTINUOUS, name='w[%s]' % i))
       # ksi = list()
        #for i in range(dimension):
            #ksi.append(model.addVar(vtype=GRB.CONTINUOUS, lb=0,name='ksi[%s]' % i))
        errorA = list()
        for i in range(m):
            errorA.append(model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='hataA[%s]' % i))
            model.update()
            model.addConstr(quicksum((A[i][j] - center[j]) * w[j] for j in range(dimension)) + ksi* ( quicksum(math.fabs(A[i][j] - center[j]) for j in range(dimension))) - gamma + 1 <= errorA[len(errorA)-1] )
            #model.addConstr(quicksum((A[i][j] - center[j]) * w[j] for j in range(dimension)) +  ( quicksum( ksi[j]*math.fabs(A[i][j] - center[j]) for j in range(dimension))) - gamma + 1 <= errorA[len(errorA)-1] )

        errorB = list()
        for i in range(p):
            errorB.append(model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='hataB[%s]' % i))
            model.update()
            model.addConstr(quicksum((B[i][j] - center[j]) * -w[j] for j in range(dimension)) -ksi*(quicksum( math.fabs(B[i][j] - center[j]) for j in range(dimension))) + gamma + 1 <= errorB[len(errorB)-1] )
            #model.addConstr(quicksum((B[i][j] - center[j]) * -w[j] for j in range(dimension)) -(quicksum( ksi[j]* math.fabs(B[i][j] - center[j]) for j in range(dimension))) + gamma + 1 <= errorB[len(errorB)-1] )

        #set objective function
        model.setObjective(c*(quicksum(i for i in errorA)/len(errorA)+cb*quicksum(i for i in errorB)/len(errorB)+ ((quicksum(i*i for i in w)+gamma+ksi)/(dimension+2))), GRB.MINIMIZE)

        #model.setObjective(c*(quicksum(i for i in errorA)/len(errorA)+quicksum(i for i in errorB)/len(errorB)+ (quicksum(i*i for i in w)+quicksum(i for i in ksi)+gamma)/(2*dimension+1)), GRB.MINIMIZE)
        #solve problem
        model.optimize()
        #get optimized parameters
        self.gamma = gamma.X
        self.ksi = ksi.X

        for i in range(dimension):
            self.w.append(w[i].X)
            #self.ksi.append(ksi[i].X)


class PCF_iterative:
    def __init__(self):
        self.pcfs = list()
        self.dimension = 0


    def fit(self, A, B,pen):
        self.dimension=len(A[0])
        while len(A) !=0 :
            center = A[random.randint(0, len(A)-1)]
            temp = PCF()
            temp.setParam(A, B,center,pen*math.sqrt(len(self.pcfs)+1),pen/math.sqrt (len(self.pcfs)+1))
            self.pcfs.append(temp)
            A = self.__updateSet(A, self.pcfs[-1], center)
            B = self.__updateSet(B, self.pcfs[-1], center)

        return self.pcfs

    def predict(self,X, labelA, labelB):
        predictions = list()
        for i in range(len(X)):
            f = 0
            for p in self.pcfs:
                f = np.dot(np.subtract(X[i], p.center), p.w) + p.ksi* np.linalg.norm( np.subtract(X[i], p.center),ord=1) - p.gamma
                #f = np.dot(np.subtract(X[i], p.center), p.w) + np.dot(p.ksi,np.absolute(np.subtract(X[i], p.center))) - p.gamma
                if f <= 0.0:
                    f = labelA
                    break
                else:
                    f = labelB

            predictions.append(f)
        return predictions

    def __delete(self,lst, indices):
        indices = set(indices)

        return [lst[i] for i in xrange(len(lst)) if i not in indices]

    def __updateSet(self, dt, pc,center):
        deleted = []

        for i in range(len(dt)):

            f = np.dot(np.subtract(dt[i], center), pc.w) + pc.ksi* np.linalg.norm(np.subtract(dt[i], pc.center), ord=1) - pc.gamma
            #f = np.dot(np.subtract(dt[i], center), pc.w) + np.dot(pc.ksi,np.absolute(np.subtract(A[i], pc.center))) - pc.gamma
            if f <= 0.0:
                deleted.append(i)
        return self.__delete(dt,deleted)


