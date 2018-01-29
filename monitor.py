import pickle
import time

import numpy as np
import tensorboard_logger as tl


def initialize_tensorboard(logdir):
    try:
        tl.configure(logdir=logdir, flush_secs=15)
        return True
    except:
        return False


def are_consistent(monitor_dual, monitor_all):
    return np.isclose(monitor_dual.get_value(), monitor_all.dual_objective)


class MonitorAllObjectives:

    def __init__(self, regularization, weights, marginals, ground_truth_centroid, trainset,
                 testset, use_tensorboard):

        self.regularization = regularization
        self.ntrain = trainset.size  # size of training set
        self.trainset = trainset
        self.ground_truth_centroid = ground_truth_centroid

        # compute the primal and dual objectives and compare them with the duality gap
        self.primal_objective = None
        self.dual_objective = None
        self.array_gaps = None
        self.duality_gap = None

        # compute the test error if a test set is present:
        self.testset = testset
        self.loss01 = None
        self.hamming = None
        if self.testset is not None:
            self.ntest = testset.size
            self.update_test_error(testset, weights)
        else:
            self.ntest = 0

        # time spent monitoring the objectives
        self.delta_time = time.time()

        self.full_batch_update(weights, marginals, count_time=False)

        # logging
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard:
            self.log_tensorboard(step=0)

        # save values in a dictionary
        self.results = {
            "number of training samples": self.ntrain,
            "number of test samples": self.ntest,
            "steps": [],
            "times": [],
            "primal objectives": [],
            "dual objectives": [],
            "duality gaps": [],
            "0/1 loss": [],
            "hamming loss": []
        }
        self.append_results()

    def __repr__(self):
        return "regularization: %f \t nb train: %i \n" \
               "Primal - Dual = %f - %f = %f \t Duality gap =  %f" \
               "Test error 0/1 = %f \t Hamming = %f" \
               % (self.regularization, self.ntrain,
                  self.primal_objective, self.dual_objective, self.primal_objective -
                  self.dual_objective, self.duality_gap,
                  self.loss01, self.hamming)

    def full_batch_update(self, weights, marginals, step, count_time):
        t1 = time.time()
        gaps_array = self.update_objectives(weights, marginals)
        t2 = time.time()
        if not count_time:  # in case of sampler update
            self.delta_time += t2 - t1

        self.update_test_error(weights)
        self.delta_time += time.time() - t2

        self.append_results(step)
        self.log_tensorboard(step)

        assert self.check_duality_gap(), self

        return gaps_array

    def check_duality_gap(self):
        """Check that the objective values are coherent with each other."""
        return np.isclose(self.duality_gap, self.primal_objective - self.dual_objective)

    def update_objectives(self, weights, marginals):
        weights_squared_norm = weights.squared_norm()

        entropies = np.array([margs.entropy() for margs in marginals])
        self.dual_objective = entropies.mean() - self.regularization / 2 * weights_squared_norm

        gaps_array = np.empty(self.ntrain, dtype=float)
        sum_log_partitions = 0
        for i, (point, margs) in enumerate(zip(self.trainset.points, marginals)):
            newmargs, log_partition = weights.infer_probabilities(point)
            gaps_array[i] = margs.kullback_leibler(newmargs)
            sum_log_partitions += log_partition

        # update the value of the duality gap
        self.duality_gap = gaps_array.mean()

        # calculate the primal score
        # take care that the log-partitions are not based on the corrected features (ground
        # truth minus feature) but on the raw features.
        self.primal_objective = \
            self.regularization / 2 * weights_squared_norm \
            + sum_log_partitions / self.ntrain \
            - weights.inner_product(self.ground_truth_centroid)

        return gaps_array

    def update_test_error(self, weights):
        if self.testset is None:
            return

        self.loss01 = 0
        self.hamming = 0
        total_labels = 0

        for point, label in self.testset:
            prediction = weights.predict(point)[0]
            tmp = np.sum(label != prediction)
            self.hamming += tmp
            self.loss01 += (tmp > 0)
            total_labels += len(label)

        self.loss01 /= self.testset.size
        self.hamming /= total_labels

    def append_results(self, step):
        self.results["steps"].append(step)
        self.results["times"].append(time.time() - self.delta_time)
        self.results["primal objectives"].append(self.primal_objective)
        self.results["dual objectives"].append(self.dual_objective)
        self.results["duality gaps"].append(self.duality_gap)
        if self.testset is not None:
            self.results["0/1 loss"].append(self.loss01)
            self.results["hamming loss"].append(self.hamming)

    def save_results(self, logdir):
        with open(logdir + "/objectives.pkl"):
            pickle.dump(self.results)

    def log_tensorboard(self, step):
        if self.use_tensorboard:
            tl.log_value("log10 duality gap", np.log10(self.duality_gap), step)
            tl.log_value("primal objective", self.primal_objective, step)
            tl.log_value("dual objective", self.dual_objective, step)
            if self.testset is not None:
                tl.log_value("01 loss", self.loss01, step)
                tl.log_value("hamming loss", self.hamming, step)

    # def log_tensorboard(self, step):
    #     if self.tensorboard and step % 10 == 0:
    #         tl.log_value("weights_squared_norm", self.weights_squared_norm, step)
    #         tl.log_value("dual objective", self.dual_objective, step)
    #         tl.log_value("log10 duality gap estimate", np.log10(self.duality_gap_estimate), step)
    #         # tl.log_value("normalized weights dot primaldir", similarity, step)
    #         # tl.log_value("step size", gammaopt, step)
    #         # tl.log_value("number of line search step", len(subobjective), step)
    #         # tl.log_value("log10 individual gap", np.log10(divergence_gap), step)
    #         # tl.log_value("log10 primaldir_squared_norm", np.log10(primaldir_squared_norm),
    #         #              step=step)
    #
    # def append_results(self):
    #     pass
    #     # TODO
    #     # if _debug and t % 10 == 0:
    #     #     # Append relevant variables
    #     #     annex.append([
    #     #         np.log10(primaldir_squared_norm),
    #     #         weights_squared_norm,
    #     #         similarity,
    #     #         dual_objective,
    #     #         np.log10(duality_gap_estimate),
    #     #         np.log10(divergence_gap),
    #     #         gammaopt,
    #     #         i,
    #     #         len(subobjective),
    #     #         t
    #     #     ])
    #
    # def save(self):
    #     # TODO
    #     pass


class MonitorDualObjective:

    def __init__(self, regularization, weights, marginals):
        self.ntrain = len(marginals)
        self.regularization = regularization
        self.entropies = np.array([margs.entropy() for margs in marginals])
        self.entropy = self.entropies.mean()
        self.weights_squared_norm = weights.squared_norm()

        self.dual_objective = self.entropy - self.regularization / 2 * self.weights_squared_norm

    def update(self, i, newmarg_entropy, norm_update):
        self.weights_squared_norm += norm_update

        tmp = self.entropies[i]
        self.entropies[i] = newmarg_entropy

        self.dual_objective += \
            (self.entropies[i] - tmp) / self.ntrain - self.regularization / 2 * norm_update

    def get_value(self):
        return self.dual_objective

    def log_tensorboard(self, step):
        tl.log_value("weights_squared_norm", self.weights_squared_norm, step)
        tl.log_value("entropy", self.entropy)
        tl.log_value("dual objective", self.dual_objective, step)


class MonitorDualityGapEstimate:

    def __init__(self, gaps_array):
        self.ntrain = len(gaps_array)
        self.gaps_array = gaps_array
        self.gap_estimate = gaps_array.mean()

    def update(self, i, new_gap):
        self.gap_estimate += (new_gap - self.gaps_array[i]) / self.ntrain
        self.gaps_array[i] = new_gap

    def get_value(self):
        return self.gap_estimate

    def log_tensorboard(self, step):
        tl.log_value("log10 duality gap estimate", np.log10(self.gap_estimate), step)
