import math
import torch as t
import torch.nn as nn
import pytorch_lightning as pl
from torch.distributions import Normal
from .abstract_bnn import AbstractLinear, AbstractConv2d

"""
Each module takes input:
  input
And produces two outputs:
  output, logPQw

Can configure:
  post_std_init
  post_std_fixed
"""


class DetParam(pl.LightningModule):
    def __init__(self, in_features, out_features, unit_std=True, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.std      = 1. if     unit_std else 1./math.sqrt(in_features)
        self.out_mult = 1. if not unit_std else 1./math.sqrt(in_features)
        self.weights = nn.Parameter(self.std * t.randn(out_features, in_features))

    def forward(self, Xi):
        S = Xi.shape[0]

        w = self.weights.expand(S, *self.weights.shape)
        Pw = Normal(0., self.std)
        _range = [*range(1, len(w.shape))]
        logP = Pw.log_prob(w).sum(_range)
        assert logP.shape == t.Size([S])
        return self.out_mult*w, logP

    def inducing_init(self, init):
        pass


class DetLinearWeights(DetParam):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__(in_features+bias, out_features, **kwargs)


class DetConv2dWeights(DetParam):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels*kernel_size**2, out_channels, **kwargs)


class DetLinear(AbstractLinear):
    Weights = DetLinearWeights


class DetConv2d(AbstractConv2d):
    Weights = DetConv2dWeights
