import numpy as np
from scipy import stats
import Utils
import Optimizor
from multiprocessing import Pool
import Coreset
import copy



class MainProgram(object):
    def __init__(self, file_path, problem_type, Z, LAMBDA=1, streaming=False):
        self.file_path = file_path
        self.pool = Pool(Utils.THREAD_NUM)
        self.coresets = [Coreset.Coreset() for i in range(Utils.REPS)]
        self.samplingProcedures = \
            (lambda i, sensitivity, sample_size, random_state=0, is_uniform=False:
             self.coresets[i].sampleCoreset(sensitivity, sample_size, random_state)) \
            if not streaming else 1

        Utils.initializaVariables(problem_type, Z, LAMBDA)
        self.P = Utils.readRealData(file_path)
        self.sample_sizes = Utils.generateSampleSizes(Utils.NUM_SAMPLES)
        self.sensitivity = None

    def computeRelativeErrorPerCoreset(self, P, time_taken):
        pass

    @classmethod
    def updateProblem(cls, file_path, problem_type, Z, LAMBDA=1, streaming=False):
        cls.pool.close()
        cls.pool.join()
        cls.pool = Pool(Utils.THREAD_NUM)
        cls.file_path = file_path
        Utils.initializaVariables(problem_type, Z, LAMBDA)
        cls.P = Utils.readRealData(cls.file_path)
        cls.samplingProcedures = \
            (lambda i, sensitivity, sample_size, random_state=0, is_uniform=False:
             cls.coresets[i].sampleCoreset(sensitivity, sample_size, random_state)) \
                if not streaming else 1

    def computeAverageEpsAndDelta(self, sensitivity, sample_size):
        func = lambda x: self.computeRelativeErrorPerCoreset(x[0], x[1])
        if Utils.PARALLELIZE:  # if parallel computation is enabled
            coresets = self.pool.map(lambda i: self.samplingProcedures(i, sensitivity, sample_size), range(Utils.REPS))
            relative_errors_and_time = self.pool.map(func, coresets)
        else:  # otherwise
            coresets = [self.samplingProcedures(i, sensitivity, sample_size) for i in range(Utils.REPS)]
            relative_errors_and_time = [func(x) for x in coresets]

        # return mean of error,
        #        mean of time,
        #        median absolute deviation of errors,
        #        median absolute deviation of times,
        #        mean coreset size
        return np.mean([x[0] for x in relative_errors_and_time]), \
               np.mean([x[1] for x in relative_errors_and_time]), \
               stats.median_absolute_deviation(np.array([x[0] for x in relative_errors_and_time])), \
               stats.median_absolute_deviation(np.array([x[0] for x in relative_errors_and_time])), \
               np.mean([x[0].d for x in relative_errors_and_time])

    def applyComaprison(self):
        mean_of_error = np.empty(Utils.NUM_SAMPLES, 2)
        mean_of_time = copy.deepcopy(mean_of_error)
        std_of_error = copy.deepcopy(mean_of_error)
        std_of_time = copy.deepcopy(mean_of_error)
        mean_of_coreset_size = copy.deepcopy(mean_of_error)
        all_sensitivities = np.vstack((np.ones(self.sensitivity.shape), self.sensitivity))

        for i in range(Utils.NUM_SAMPLES):
            sample_size = self.sample_sizes[i]
            for type_of_sampling_alg in range(2):
                mean_of_error[i, type_of_sampling_alg], mean_of_time[i, type_of_sampling_alg],\
                    std_of_error[i, type_of_sampling_alg], std_of_time[i, type_of_sampling_alg], \
                    mean_of_time[i, type_of_sampling_alg] = \
                    self.computeAverageEpsAndDelta(all_sensitivities, sample_size)

        np.savez('results.npz', mean_of_error=mean_of_error, mean_of_time=mean_of_time, std_of_error=std_of_error,
                 std_of_time=std_of_time, mean_of_coreset_size=mean_of_coreset_size)

