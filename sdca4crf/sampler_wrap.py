import numpy as np

from sdca4crf.parameters.radius import radii
from .sampler import Sampler


class SamplerWrap:
    UNIFORM = 0
    IMPORTANCE = 1
    GAP = 2
    GAPP = 3
    MAX = 4

    def __init__(self, sampling_scheme, non_uniformity,
                 gaps_array, trainset, regularization):

        self.size = len(trainset)
        self.non_uniformity = non_uniformity

        if sampling_scheme == "uniform":
            self.scheme = SamplerWrap.UNIFORM
        elif sampling_scheme == "importance":
            self.scheme = SamplerWrap.IMPORTANCE
        elif sampling_scheme == "gap":
            self.scheme = SamplerWrap.GAP
        elif sampling_scheme == "gap+":
            self.scheme = SamplerWrap.GAPP
        elif sampling_scheme == "max":
            self.scheme = SamplerWrap.MAX
        else:
            raise ValueError(" %s is not a valid argument for sampling scheme" % str(
                sampling_scheme))

        if self.scheme in [SamplerWrap.UNIFORM, SamplerWrap.GAP, SamplerWrap.MAX]:
            self.importances = np.ones(self.size)
        elif self.scheme in [SamplerWrap.IMPORTANCE, SamplerWrap.GAPP]:
            self.importances = 1 + radii(trainset) ** 2 / self.size / regularization

        self.sampler = Sampler(gaps_array * self.importances,
                               is_determinist=(self.scheme == SamplerWrap.MAX))

    def update(self, sample_id, divergence_gap):
        if self.scheme in [SamplerWrap.GAP, SamplerWrap.GAPP]:
            self.sampler.update(divergence_gap * self.importances[sample_id], sample_id)

    def full_update(self, gaps_array):
        if self.scheme in [SamplerWrap.GAP, SamplerWrap.GAPP]:
            self.sampler = Sampler(gaps_array * self.importances)

    def sample(self):
        if np.random.rand() > self.non_uniformity:  # then sample uniformly
            return np.random.randint(self.size)
        else:  # sample proportionally to the duality gaps
            return self.sampler.sample()
