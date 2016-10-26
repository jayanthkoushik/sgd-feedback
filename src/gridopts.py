import itertools

import numpy as np
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax

from dna import DNA
from eveprop import Eveprop


class GridOptimizer:

    def __init__(self, optimizer, param_vals):
        self.optimizer = optimizer
        self.grid = list(dict(zip(param_vals.keys(), x)) for x in itertools.product(*param_vals.values()))

    def __iter__(self):
        self.grid_iter = iter(self.grid)
        return self

    def __next__(self):
        return self.optimizer(**next(self.grid_iter))


class GridSGDMomentum(GridOptimizer):

    def __init__(self, lrs, momentums, decays):
        super().__init__(SGD, {"lr": lrs, "momentum": momentums, "decay": decays})


class GridSGDNesterov(GridOptimizer):

    def __init__(self, lrs, momentums, decays):
        super().__init__(SGD, {"lr": lrs, "momentum": momentums, "decay": decays, "nesterov": [True]})


class GridRMSprop(GridOptimizer):

    def __init__(self, lrs, decays):
        if not any(np.isclose(lrs, 0.001)):
            lrs = list(lrs) + [0.001]
        super().__init__(RMSprop, {"lr": lrs, "decay": decays})


class GridAdagrad(GridOptimizer):

    def __init__(self, lrs, decays):
        if not any(np.isclose(lrs, 0.01)):
            lrs = list(lrs) + [0.01]
        super().__init__(Adagrad, {"lr": lrs, "decay": decays})


class GridAdadelta(GridOptimizer):

    def __init__(self, lrs, decays):
        if not any(np.isclose(lrs, 1.0)):
            lrs = list(lrs) + [1.0]
        super().__init__(Adadelta, {"lr": lrs, "decay": decays})


class GridAdam(GridOptimizer):

    def __init__(self, lrs, decays):
        if not any(np.isclose(lrs, 0.001)):
            lrs = list(lrs) + [0.001]
        super().__init__(Adam, {"lr": lrs, "decay": decays})


class GridAdamax(GridOptimizer):

    def __init__(self, lrs, decays):
        if not any(np.isclose(lrs, 0.002)):
            lrs = list(lrs) + [0.002]
        super().__init__(Adamax, {"lr": lrs, "decay": decays})


class GridDNA(GridOptimizer):

    def __init__(self, lrs, decays):
        if not any(np.isclose(lrs, 0.0001)):
            lrs = list(lrs) + [0.0001]
        super().__init__(DNA, {"lr": lrs, "decay": decays})


class GridEveprop(GridOptimizer):

    def __init__(self, lrs, decays):
        if not any(np.isclose(lrs, 0.001)):
            lrs = list(lrs) + [0.001]
        super().__init__(Eveprop, {"lr": lrs, "decay": decays})


OPTIMIZERS_INDEX = {
    "sgdmomentum": GridSGDMomentum,
    "sgdnesterov": GridSGDNesterov,
    "rmsprop": GridRMSprop,
    "adagrad": GridAdagrad,
    "adadelta": GridAdadelta,
    "adam": GridAdam,
    "adamax": GridAdamax,
    "dna": GridDNA,
    "eveprop": GridEveprop
}

