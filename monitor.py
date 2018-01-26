import pickle
import time

import numpy as np
import tensorboard_logger as tl


class Monitor:

    def __init__(self, regularization, ntrain, npass, logdir):
        self.regularization = regularization
        self.ntrain = ntrain  # size of training set
        self.npass = npass  # maximum number of epochs

        # log with tensorboard logger
        if logdir is not None:
            try:
                tl.configure(logdir=logdir, flush_secs=15)
                self.tensorboard = True
            except:
                self.tensorboard = False


class MonitorEpoch(Monitor):

    def __init__(self, regularization, x, ground_truth_centroid, weights, marginals,
                 npass, sampler_period, xtest=None, ytest=None, logdir=None):
        Monitor.__init__(self, regularization, len(x), npass, logdir)

        self.sampler_period = sampler_period

        # compute the primal and dual objectives and compare them with the duality gap
        self.primal_objective = None
        self.dual_objective = None
        self.array_gaps = None
        self.duality_gap = None
        self.delta_time = time.time()
        self.full_batch_update(marginals, weights, x, ground_truth_centroid)

        # compute the test error if a test set is present:
        self.do_test = xtest is not None and ytest is not None
        self.loss01 = None
        self.hamming = None
        if self.do_test:
            self.ntest = len(ytest)
            self.test_error(xtest, ytest, weights)
        else:
            self.ntest = 0

        self.logdir = logdir

        # log with tensorboard logger
        if self.logdir is not None:
            try:
                tl.configure(logdir=logdir, flush_secs=15)
                self.tensorboard = True
            except:
                self.tensorboard = False

        if self.tensorboard:
            self.log_tensorboard(0)

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
               "\|w\|^2= %f \t entropy = %f \n" \
               "Primal - Dual = %f - %f = %f \t Duality gap =  %f" \
               "Test error 0/1 = %f \t Hamming = %f" \
               % (self.regularization, self.ntrain, self.weights_squared_norm, self.entropy,
                  self.primal_objective, self.dual_objective, self.primal_objective -
                  self.dual_objective, self.duality_gap,
                  self.loss01, self.hamming)

    def full_batch_update(self, marginals, weights, x, ground_truth_centroid):
        t1 = time.time()

        weights_squared_norm = weights.squared_norm()

        entropies = np.array([margs.entropy() for margs in marginals])
        self.dual_objective = entropies.mean() - self.regularization / 2 * weights_squared_norm

        array_gaps = np.empty(self.ntrain, dtype=float)
        sum_log_partitions = 0
        for i, (margs, imgs) in enumerate(zip(marginals, x)):
            newmargs, log_partition = weights.infer_probabilities(imgs)
            array_gaps[i] = margs.kullback_leibler(newmargs)
            sum_log_partitions += log_partition

        # update the value of the duality gap
        self.duality_gap = array_gaps.mean()

        # calculate the primal score
        # take care that the log-partitions are not based on the corrected features (ground
        # truth minus feature) but on the raw features.
        self.primal_objective = \
            self.regularization / 2 * weights_squared_norm \
            + sum_log_partitions / self.ntrain \
            - weights.inner_product(ground_truth_centroid)

        self.delta_time = time.time() - t1

        # check that this is coherent with my values
        # of the duality gap and the dual objective
        assert self.check_norm(), self
        assert self.check_duality_gap(), self

        return array_gaps

    # def check_norm(self, weights):
    #     return np.isclose(self.weights_squared_norm, weights.squared_norm())

    def check_duality_gap(self):
        return np.isclose(self.duality_gap, self.primal_objective - self.dual_objective)

    def test_error(self, xtest, ytest, weights):
        self.loss01, self.hamming = weights.prediction_loss(xtest, ytest)

    def append_results(self, step):
        self.results["steps"].append(step)
        self.results["times"].append(time.time() - self.delta_time)
        self.results["primal objectives"].append(self.primal_objective)
        self.results["dual objectives"].append(self.dual_objective)
        self.results["duality gaps"].append(self.duality_gap)
        if self.do_test:
            self.results["0/1 loss"].append(self.loss01)
            self.results["hamming loss"].append(self.hamming)

    def save_results(self):
        if self.logdir:
            with open(self.logdir + "/objectives.pkl"):
                pickle.dump(self.results)

    def log_tensorboard(self, step):
        if self.tensorboard:
            tl.log_value("log10 duality gap", np.log10(self.duality_gap), step)
            tl.log_value("primal objective", self.primal_objective, step)
            tl.log_value("dual objective", self.dual_objective, step)
            if self.do_test:
                tl.log_value("01 loss", self.loss01, step)
                tl.log_value("hamming loss", self.loss_hamming, step)


class MonitorIteration(Monitor):

    def __init__(self, regularization, x, weights_squared_norm, marginals,
                 npass, array_gaps, logdir=None):
        Monitor.__init__(self, regularization, len(x), npass, logdir)

        self.weights_squared_norm = weights_squared_norm
        self.entropies = np.array([margs.entropy() for margs in marginals])
        self.entropy = self.entropies.mean()
        self.dual_objective = self.entropy - self.regularization / 2 * self.weights_squared_norm

        self.array_gaps = array_gaps
        self.duality_gap_estimate = self.array_gaps.mean()

    def update_gap_estimate(self, value):
        self.duality_gap_estimate = value

    def frequent_update(self, i, newmarginal, gammaopt, weights_dot_primaldir,
                        primaldir_squared_norm, divergence_gap):
        # update weights norm, entropy  and dual objective
        norm_update = \
            gammaopt * 2 * weights_dot_primaldir \
            + gammaopt ** 2 * primaldir_squared_norm
        self.weights_squared_norm += norm_update

        tmp = self.entropies[i]
        self.entropies[i] = newmarginal.entropy()

        self.dual_objective += \
            (self.entropies[i] - tmp) / self.ntrain - self.regularization / 2 * norm_update

        similarity = weights_dot_primaldir / np.sqrt(primaldir_squared_norm) / np.sqrt(
            self.weights_squared_norm)

        self.duality_gap_estimate += (divergence_gap - self.array_gaps[i]) / self.ntrain
        self.array_gaps[i] = divergence_gap

    def log_frequent_tensorboard(self, step):
        if self.tensorboard and step % 10 == 0:
            tl.log_value("weights_squared_norm", self.weights_squared_norm, step)
            tl.log_value("dual objective", self.dual_objective, step)
            tl.log_value("log10 duality gap estimate", np.log10(self.duality_gap_estimate), step)
            # tl.log_value("normalized weights dot primaldir", similarity, step)
            # tl.log_value("step size", gammaopt, step)
            # tl.log_value("number of line search step", len(subobjective), step)
            # tl.log_value("log10 individual gap", np.log10(divergence_gap), step)
            # tl.log_value("log10 primaldir_squared_norm", np.log10(primaldir_squared_norm),
            #              step=step)

    def append_results(self):
        pass
        # TODO
        # if _debug and t % 10 == 0:
        #     # Append relevant variables
        #     annex.append([
        #         np.log10(primaldir_squared_norm),
        #         weights_squared_norm,
        #         similarity,
        #         dual_objective,
        #         np.log10(duality_gap_estimate),
        #         np.log10(divergence_gap),
        #         gammaopt,
        #         i,
        #         len(subobjective),
        #         t
        #     ])

    def save(self):
        # TODO
        pass
