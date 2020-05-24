import numpy as np
import PointSet
# from MainProgram.Utils import SENSE_BOUND
import MainProgram


class Coreset(PointSet.PointSet):
    def __init__(self, P, W):
        super().__init__(self, P, W, MainProgram.Utils.ELLIPSOID_MAX_ITER)

        self.P = None
        self.rank = 0
        self.sensitivities = None
        self._SENSE_BOUND = MainProgram.Utils.SENSE_BOUND
        return

    def computeCoreset(self, P, sample_size):
        pass

    def computeSensitivity(self, P, use_svd=True, sum_old_weights=None):
        P.computeU(use_svd)
        sensitivitiy = np.empty(P.n, )

        if sum_old_weights is None:
            sum_old_weights = np.sum(P.W)

        for i in range(P.n):
            sensitivitiy[i] = self._SENSE_BOUND(P.W)

        return sensitivitiy




    def mergeCoresets(self, coreset, sample_size):
        self.P.mergePointSet(coreset.P)
        self.computeCoreset(self.P, sample_size)
        self.rank += 1