import json
import sys
import os
import pickle
import tarfile
from argparse import ArgumentParser

import numpy as np
from keras.preprocessing import sequence
from keras.datasets import mnist, cifar10, cifar100, imdb
from keras.utils import np_utils
from keras.utils.data_utils import get_file

from gridopts import *
from models import *
from dna import DNAMonitor
from babi_sitter import *

def pre_process_image(args):
    (X_train, y_train), _ = DATASET_INFO[args.dataset]["loader"]()
    if args.n_samples is not None:
        p = np.random.permutation(X_train.shape[0])
        X_train, y_train = X_train[p][:args.n_samples], y_train[p][:args.n_samples]
    X_train = X_train.astype("float32") / 255.
    if X_train.ndim == 3:
        X_train = X_train[:, np.newaxis, :, :]
    if args.flatten:
        X_train = X_train.reshape((X_train.shape[0], -1))
    y_train = np_utils.to_categorical(y_train, DATASET_INFO[args.dataset]["nb_classes"])
    return X_train, y_train


def pre_process_text(args):
    (X_train, y_train), _ = DATASET_INFO[args.dataset]["loader"](nb_words=args.n_vocab)
    if args.n_samples is not None:
        X_train, y_train = X_train[:args.n_samples], y_train[:args.n_samples]
    X_train = sequence.pad_sequences(X_train, maxlen=args.max_len)
    y_train = np.array(y_train)
    return X_train, y_train


def pre_process_babi(args):
    try:
        path = get_file('babi-tasks-v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
    except:
        print('Error downloading dataset, please download it manually:\n'
              '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
              '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
        raise
    tar = tarfile.open(path)
    train = get_stories(tar.extractfile(QFILE[args.babi_qnum]))
    vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train)))
    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    story_maxlen = max(map(len, (x for x, _, _ in train)))
    query_maxlen= max(map(len, (x for _, x, _ in train)))

    X, Xq, Y = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
    return [X, Xq], Y, vocab_size, story_maxlen, query_maxlen


DATASET_INFO = {
    "mnist": {"loader": mnist.load_data, "nb_classes": 10, "preprocess": pre_process_image},
    "cifar10": {"loader": cifar10.load_data, "nb_classes": 10, "preprocess": pre_process_image},
    "cifar100": {"loader": cifar100.load_data, "nb_classes": 100, "preprocess": pre_process_image},
    "imdb": {"loader": imdb.load_data, "nb_classes": 2, "preprocess": pre_process_text},
    "babi": {"loader": imdb.load_data, "preprocess": pre_process_babi}
}


arg_parser = ArgumentParser()
arg_parser.add_argument("--optimizer", type=str, required=True, choices=OPTIMIZERS_INDEX.keys())
arg_parser.add_argument("--opt-args", type=json.loads, required=True)
arg_parser.add_argument("--model", type=str, required=True, choices=MODEL_FACTORIES.keys())
arg_parser.add_argument("--dataset", type=str, required=True, choices=DATASET_INFO.keys())
arg_parser.add_argument("--batch-size", type=int, required=True)
arg_parser.add_argument("--epochs", type=int, required=True)
arg_parser.add_argument("--save-path", type=str, required=True)
arg_parser.add_argument("--n-samples", type=int, default=None)
arg_parser.add_argument("--max-len", type=int, default=100)  # text hyperparam.
arg_parser.add_argument("--n-vocab", type=int, default=20000)  # text hyperparam.
arg_parser.add_argument("--embed-dim", type=int, default=256)  # text hyperparam.
arg_parser.add_argument("--hidden-dim", type=int, default=256)  # text hyperparam.
arg_parser.add_argument("--babi-qnum", type=int, default=1)  # babi hyperparam.
arg_parser.add_argument("--flatten", action="store_true")
args = arg_parser.parse_args()

try:
    open(args.save_path, "w+")
except FileNotFoundError:
    print("Error: save path is not accessible")
    sys.exit(1)
os.remove(args.save_path)

grid_opt = OPTIMIZERS_INDEX[args.optimizer](**args.opt_args)

if args.dataset == "babi":
    X_train, y_train, vocab_size, story_maxlen, query_maxlen = DATASET_INFO[args.dataset]["preprocess"](args)
else:
    X_train, y_train = DATASET_INFO[args.dataset]["preprocess"](args)

best_final_loss = np.inf
for opt in grid_opt:
    if args.model == 'bigru':
        model = MODEL_FACTORIES[args.model](args)
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    elif args.model == "babi_gru":
        model = MODEL_FACTORIES[args.model](args.embed_dim, vocab_size, story_maxlen, query_maxlen)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    else:
        model = MODEL_FACTORIES[args.model](X_train.shape[1:], DATASET_INFO[args.dataset]["nb_classes"])
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    if args.optimizer == "dna" or args.optimizer == "eveprop":
        dna_monitor = DNAMonitor()
        callbacks = [dna_monitor]
    else:
        callbacks = []
    history = model.fit(x=X_train, y=y_train, batch_size=args.batch_size, nb_epoch=args.epochs, verbose=1, callbacks=callbacks)

    if history.history["loss"][-1] < best_final_loss:
        best_final_loss = history.history["loss"][-1]
        best_loss_history = history.history["loss"]
        best_opt_config = opt.get_config()
        best_decay = opt.decay.get_value()
        if args.optimizer == "dna" or args.optimizer == "eveprop":
            best_dna_monitor = dna_monitor

save_data = {
    "best_loss_history": best_loss_history,
    "param_grid": grid_opt.grid,
    "best_opt_config": best_opt_config,
    "best_decay": best_decay,
    "cmd_args": args,
}
if args.optimizer == "dna" or args.optimizer == "eveprop":
    save_data["best_batch_loss_history"] = best_dna_monitor.batch_losses
    save_data["ds"] = best_dna_monitor.ds
with open(args.save_path, "wb") as f:
    pickle.dump(save_data, f)
