import itertools

import numpy as np
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax

from dna import DNA


class GridOptimizer:

    def __init__(self, optimizer, param_vals):
        self.optimizer = optimizer
        self.grid = list(dict(zip(param_vals.keys(), x)) for x in itertools.product(*param_vals.values()))

    def __iter__(self):
        self.grid_iter = iter(self.grid)
        return self

    def __next__(self):
        return self.optimizer(**next(self.grid_iter))


class GridSGD(GridOptimizer):

    def __init__(self, lrs):
        super().__init__(SGD, {"lr": lrs})


class GridSGDMomentum(GridOptimizer):

    def __init__(self, lrs, momentums):
        super().__init__(SGD, {"lr": lrs, "momentum": momentums})


class GridSGDNesterov(GridOptimizer):

    def __init__(self, lrs, momentums):
        super().__init__(SGD, {"lr": lrs, "momentum": momentums, "nesterov": [True]})


class GridRMSprop(GridOptimizer):

    def __init__(self, lrs):
        if not any(np.isclose(lrs, 0.001)):
            lrs = list(lrs) + [0.001]
        super().__init__(RMSprop, {"lr": lrs})


class GridAdagrad(GridOptimizer):

    def __init__(self, lrs):
        if not any(np.isclose(lrs, 0.01)):
            lrs = list(lrs) + [0.01]
        super().__init__(Adagrad, {"lr": lrs})


class GridAdadelta(GridOptimizer):

    def __init__(self, lrs):
        if not any(np.isclose(lrs, 1.0)):
            lrs = list(lrs) + [1.0]
        super().__init__(Adadelta, {"lr": lrs})


class GridAdam(GridOptimizer):

    def __init__(self, lrs):
        if not any(np.isclose(lrs, 0.001)):
            lrs = list(lrs) + [0.001]
        super().__init__(Adam, {"lr": lrs})


class GridAdamax(GridOptimizer):

    def __init__(self, lrs):
        if not any(np.isclose(lrs, 0.002)):
            lrs = list(lrs) + [0.002]
        super().__init__(Adamax, {"lr": lrs})


class GridDNA(GridOptimizer):

    def __init__(self, lrs):
        if not any(np.isclose(lrs, 0.001)):
            lrs = list(lrs) + [0.001]
        super().__init__(DNA, {"lr": lrs})


OPTIMIZERS_INDEX = {
    "sgd": GridSGD,
    "sgdmomentum": GridSGDMomentum,
    "sgdnesterov": GridSGDNesterov,
    "rmsprop": GridRMSprop,
    "adagrad": GridAdagrad,
    "adadelta": GridAdadelta,
    "adam": GridAdam,
    "adamax": GridAdamax,
    "dna": GridDNA
}

