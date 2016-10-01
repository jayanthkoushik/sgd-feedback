import json
import pickle
import os
from argparse import ArgumentParser

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from keras.datasets import mnist
from keras.utils import np_utils

from gridopts import *

LRS = np.logspace(-6, -1, 10)
BATCH_SIZE = 128
EPOCHS = 200

arg_parser = ArgumentParser()
arg_parser.add_argument("--optimizer", type=str, required=True, choices=OPTIMIZERS_INDEX.keys())
arg_parser.add_argument("--opt-args", type=json.loads, default={})
args = arg_parser.parse_args()

args.opt_args["lrs"] = LRS
grid_opt = OPTIMIZERS_INDEX[args.optimizer](**args.opt_args)

(X_train, y_train), _ = mnist.load_data()
X_train = X_train.reshape((X_train.shape[0], -1)).astype("float32") / 255.
y_train = np_utils.to_categorical(y_train, 10)

best_final_loss = np.inf
for opt in grid_opt:
    model = Sequential()
    model.add(Dense(1000, activation="relu", W_regularizer=l2(0.0001), input_shape=(784, )))
    model.add(Dense(1000, activation="relu", W_regularizer=l2(0.0001)))
    model.add(Dense(10, activation="softmax", W_regularizer=l2(0.0001)))

    model.compile(optimizer=opt, loss="categorical_crossentropy")
    history = model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, verbose=1)

    if history.history["loss"][-1] < best_final_loss:
        best_final_loss = history.history["loss"][-1]
        best_loss_history = history.history["loss"]
        best_opt_config = opt.get_config()

save_data = {"best_loss_history": best_loss_history, "param_grid": grid_opt.grid, "best_opt_config": best_opt_config}
with open(os.path.join("data", "mlnnexp", "{}.pkl".format(args.optimizer)), "wb") as f:
    pickle.dump(save_data, f)

