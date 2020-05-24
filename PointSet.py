import numpy as np
import MVEEApprox


class PointSet(object):
    def __init__(self, P, W=None, ellipsoid_max_iters=10):
        self.P = P
        self.n, self.d = P.shape
        self.W = np.ones((self.n, )) if W is None else W
        self.U = self.D = self.V = None
        self.mvee = None
        self.ellipsoid_max_iters = ellipsoid_max_iters
        self.cost_func = lambda x: np.linalg.norm(np.multiply(self.W, np.dot(self.P, x), ord=1))
        self.computeU()

    def computeU(self, use_svd=True):
        if use_svd:
            self.U, self.D, self.V = np.linalg.svd(np.multiply(np.sqrt(self.W), self.P), full_matrices=False)
        else:
            self.mvee = MVEEApprox.MVEEApprox(self.P, self.cost_func, self.ellipsoid_max_iters)
            ellipsoid, _ = self.mvee.compute_approximated_MVEE()
            _, self.D, self.V = np.linalg.svd(np.linalg.pinv(ellipsoid), full_matrices=True)
            self.U = np.dot(self.P, np.linalg.inv(self.D.dot(self.V)))

    def mergePointSet(self, Q):
        self.P = np.vstack((self.P, Q.P))
        self.W = np.hstack((self.W, Q.W))
        self.n, self.d = self.P.shape
        self.computeU()

