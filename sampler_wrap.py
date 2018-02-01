import numpy as np

from sampler import Sampler


class SamplerWrap:
    UNIFORM = 0
    IMPORTANCE = 1
    GAP = 2
    GAPP = 3

    def __init__(self, sampling_scheme, gaps_array, features_cls, trainset, regularization):

        if sampling_scheme == "uniform":
            self.scheme = SamplerWrap.UNIFORM
        elif sampling_scheme == "importance":
            self.scheme = SamplerWrap.IMPORTANCE
        elif sampling_scheme == "gap":
            self.scheme = SamplerWrap.GAP
        elif sampling_scheme == "gap+":
            self.scheme = SamplerWrap.GAPP
        else:
            raise ValueError(" %s is not a valid argument for sampling scheme" % str(
                sampling_scheme))

        if self.scheme in [SamplerWrap.UNIFORM, SamplerWrap.GAP]:
            self.importances = np.ones(trainset.size)
        elif self.scheme in [SamplerWrap.IMPORTANCE, SamplerWrap.GAPP]:
            self.importances = 1 + features_cls.radii(trainset.points, trainset.labels) ** 2 \
                               / trainset.size / regularization

        self.sampler = Sampler(gaps_array * self.importances)

    def update(self, sample_id, divergence_gap):
        if self.sampling_scheme in [SamplerWrap.GAP, SamplerWrap.GAPP]:
            self.sampler.update(divergence_gap * self.importances[sample_id], sample_id)

    def full_update(self, gaps_array):
        if self.sampling_scheme in [SamplerWrap.GAP, SamplerWrap.GAPP]:
            self.sampler = Sampler(gaps_array * self.importances)
