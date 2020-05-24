import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import scipy as sp
import RegressionProblems as rp
import time
from multiprocessing import Lock


class Optimizor(object):
    MODELS = {
        'logistic': lambda C, tol, Z: LogisticRegression(tol=tol, C=C, solver='lbfgs', max_iter=1e4),
        'svm': lambda C, tol, Z: SVC(kernel='linear', C=C, tol=tol),
        'lz': lambda C, tol, Z: rp.RegressionProblem(Z)
    }

    # create mutex for multi-threading purposes
    mutex = Lock()

    def __init__(self, problem_type, C, tol, Z):
        self.model = Optimizor.MODELS[problem_type](C=C, tol=tol, Z=Z)
        self.sum_weights = None
        self.C = C

    def fit(self, P):
        start_time = time.time()
        if Utils.PROBLEM_TYPE != 'lz':
            Optimizor.mutex.acquire()
            c_prime = self.model.C * float(self.sum_weights / (np.sum(P.W)))
            params = {"C": c_prime}
            self.model.set_params(**params)
            Optimizor.mutex.release()

        self.model.fit(P.P[:, :-1], P.P[:, -1], P.W)
        Optimizor.mutex.acquire()
        w, b = self.model.coef_, self.model.intercept_
        sol = np.hstack((w, b)) if b is not None else w
        Optimizor.mutex.release()
        return self.computeCost(P, sol), time.time() - start_time

    def computeCost(self, P, x):
        return Utils.OBJECTIVE_COST(P, x, self.sum_weights)

    def defineSumOfWegiths(self, W):
        self.sum_weights = np.sum(W)
