import numpy as np
import PointSet
from MainProgram import Utils


class Coreset(PointSet):
    def __init__(self):
        self.P = None
        self.rank = 0
        return

    def computeCoreset(self, P, sample_size):
        pass

    @staticmethod
    def computeSensitivity(self, P):



    def mergeCoresets(self, coreset, sample_size):
        self.P.mergePointSet(coreset.P)
        self.computeCoreset(self.P, sample_size)
        self.rank += 1