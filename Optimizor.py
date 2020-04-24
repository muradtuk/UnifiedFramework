import numpy as np
import sklearn
import scipy as sp
from MainProgram import Utils
import RegressionProblems as rp


class Optimizor(object):
    MODELS = {
        'logistic': sklearn.linear_model.LogisticRegression(tol=Utils.OPTIMIZATION_TOL, C=Utils.LAMBDA, solver='lbfgs',
                                                             max_iter=1e4),
        'svm': sklearn.svm.SVC(kernel='linear', C=Utils.LAMBDA, tol=Utils.OPTIMIZATION_TOL),
        'lz': rp.RegressionProblem(Utils.Z)
    }

    def __init__(self):

        self.model = Optimizor.MODELS[Utils.PROBLEM_TYPE]

    @staticmethod
    def solve():
        pass