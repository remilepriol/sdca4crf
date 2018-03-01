import numpy as np

from sampler import Sampler
from weights import radii


class SamplerWrap:
    UNIFORM = 0
    IMPORTANCE = 1
    GAP = 2
    GAPP = 3

    def __init__(self, sampling_scheme, non_uniformity,
                 gaps_array, trainset, regularization):

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
            self.importances = np.ones(len(trainset))
        elif self.scheme in [SamplerWrap.IMPORTANCE, SamplerWrap.GAPP]:
            self.importances = 1 + radii(trainset) ** 2 / len(trainset) / regularization

        self.sampler = Sampler(gaps_array * self.importances)
        self.non_uniformity = non_uniformity

    def update(self, sample_id, divergence_gap):
        if self.scheme in [SamplerWrap.GAP, SamplerWrap.GAPP]:
            self.sampler.update(divergence_gap * self.importances[sample_id], sample_id)

    def full_update(self, gaps_array):
        if self.scheme in [SamplerWrap.GAP, SamplerWrap.GAPP]:
            self.sampler = Sampler(gaps_array * self.importances)

    def sample(self):
        return self.sampler.mixed_sample(self.non_uniformity)