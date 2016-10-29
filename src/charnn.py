import functools
import math
import argparse
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import theano
import theano.tensor as T
import numpy as np
from theano.tensor.extra_ops import repeat

from theano_utils import *

print = functools.partial(print, flush=True)
srng = T.shared_randomstreams.RandomStreams()
floatX = theano.config.floatX

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--data-file", type=str, required=True)
arg_parser.add_argument("--save-dir", type=str, required=True)
arg_parser.add_argument("--num-hs", type=int, required=True, nargs="+")
arg_parser.add_argument("--dropouts", type=float, required=True, nargs="+")
arg_parser.add_argument("--seq-size", type=int, required=True)
arg_parser.add_argument("--batch-size", type=int, required=True)
arg_parser.add_argument("--num-epochs", type=int, required=True)
arg_parser.add_argument("--patience", type=int, required=True)
arg_parser.add_argument("--optimizer", type=str, required=True, choices=["adam", "dna", "rmsprop"])
arg_parser.add_argument("--lr", type=float, required=True)
arg_parser.add_argument("--dec", type=float, required=True)
args = arg_parser.parse_args()

if not os.path.exists(os.path.join(args.save_dir, "samples")):
    os.makedirs(os.path.join(args.save_dir, "samples"))

with open(args.data_file) as f:
    text = f.read()

idx_chars = list(set(text))
char_idxs = {c: i for i, c in enumerate(idx_chars)}
V = len(idx_chars)
tr_len = int(0.8 * len(text))
v_len = int(0.1 * len(text))
text_splits = {"train": text[:tr_len], "val": text[tr_len:tr_len + v_len], "test": text[tr_len + v_len:]}
print("Split sizes: train {}, val {}, test {}".format(*[len(text_splits[split]) for split in ["train", "val", "test"]]))

X_splits, y_splits, X_data, y_data = {}, {}, {}, {}
for split in ["train", "val", "test"]:
    x_text = text_splits[split][:-1]
    y_text = text_splits[split][1:]

    text_len = len(x_text)
    num_seqs = math.ceil(text_len / args.seq_size)
    num_batches = math.ceil(num_seqs / args.batch_size)

    X_splits[split], y_splits[split] = [-1 * np.ones((num_seqs, args.seq_size), dtype=floatX) for _ in range(2)]

    cur_batch = 0
    cur_pos_in_batch = 0
    for seq_num in range(num_seqs):
        for z_splits, z_text in zip([X_splits, y_splits], [x_text, y_text]):
            seq = [char_idxs[c] for c in z_text[seq_num * args.seq_size : (seq_num + 1) * args.seq_size]]
            z_splits[split][cur_batch * args.batch_size + cur_pos_in_batch, :len(seq)] = seq

        cur_batch += 1
        if cur_batch == num_batches or cur_batch * args.batch_size + cur_pos_in_batch >= num_seqs:
            cur_batch = 0
            cur_pos_in_batch += 1

    X_data[split] = theano.shared(X_splits[split], borrow=True)
    y_data[split] = theano.shared(y_splits[split], borrow=True)

def W_init(size):
    return theano.shared(np.random.uniform(low=-.1, high=.1, size=size).astype(floatX), borrow=True)

def b_init(size):
    return theano.shared(np.zeros(size, dtype=floatX), borrow=True)

X = T.matrix()
y = T.matrix()

ss, Wzs, Wrs, Whs, bzs, brs, bhs = [[] for _ in range(7)]
for i, (num_h, num_h_prev) in enumerate(zip(args.num_hs, [V] + args.num_hs)):
    ss.append(theano.shared(np.zeros((args.batch_size, num_h), dtype=floatX), borrow=True))
    for Ws in [Wzs, Wrs, Whs]:
        Ws.append(W_init((num_h_prev + num_h, num_h)))
    for bs in [bzs, brs, bhs]:
        bs.append(b_init((num_h, )))

    if i == len(args.num_hs) - 1:
        Wo = W_init((num_h, V))
        bo = b_init((V, ))

train_premasks, train_postmasks = [], []
dropout_masks = []
for dropout, num_h in zip(args.dropouts, args.num_hs):
    dropout_masks.append(srng.binomial(size=(X.shape[1], X.shape[0], num_h), p=dropout))

def gru_step(xt, *dropout_maskst):
    xt_oh = T.zeros((xt.shape[0], V), dtype=floatX)
    xt_oh = T.set_subtensor(xt_oh[T.arange(xt.shape[0]), T.cast(xt, "int32")], 1)

    sts = []
    layer_in = xt_oh
    for i, num_h in enumerate(args.num_hs):
        stm1 = ss[i][:xt.shape[0], :]
        xs = T.concatenate([layer_in, stm1], axis=1)
        z = T.nnet.hard_sigmoid(T.dot(xs, Wzs[i]) + bzs[i])
        r = T.nnet.hard_sigmoid(T.dot(xs, Wrs[i]) + brs[i])
        h = T.tanh(T.dot(T.concatenate([layer_in, stm1*r], axis=1), Whs[i]) + bhs[i])

        st = (1 - z)*h + z*stm1
        st = T.set_subtensor(st[T.nonzero(T.eq(xt, -1)), :], stm1[T.nonzero(T.eq(xt, -1)), :])
        sts.append(st)

        if dropout_maskst and args.dropouts[i] != 0:
            st = dropout_layer(st, args.dropouts[i], srng, train_premasks, train_postmasks, dropout_maskst[i])

        if i == len(args.num_hs) - 1:
            ot = T.nnet.softmax(T.dot(st, Wo) + bo)
        else:
            layer_in = st

    xtp1 = T.cast(T.argmax(srng.multinomial(n=1, pvals=ot), axis=1), floatX)

    s_updates = OrderedDict()
    for s, st, num_h in zip(ss, sts, args.num_hs):
        if T.lt(xt.shape[0], s.shape[0]):
            pad = T.zeros((s.shape[0] - xt.shape[0], num_h), dtype=floatX)
            st = T.concatenate([st, pad], axis=0)
        s_updates[s] = st
    return [ot, xtp1], s_updates

[o, _], gru_train_updates = theano.scan(gru_step, outputs_info=[None, None], sequences=[X.T] + dropout_masks)
o = o.dimshuffle((1, 0, 2))
p_hat = o[repeat(T.arange(o.shape[0]).dimshuffle(0, "x"), o.shape[1], axis=1), repeat(T.arange(o.shape[1]).dimshuffle("x", 0), o.shape[0], axis=0), T.cast(y, "int32")]
y_mask = T.neq(y, -1)  # Evolves into c_fagrigus
cross_entropy = -T.mean(T.sum(T.log(p_hat) * y_mask, axis=1) / T.sum(y_mask, axis=1))

def perplexity(y, o):
    p_hat = o[np.repeat(np.arange(o.shape[0]).reshape((-1, 1)), o.shape[1], axis=1), np.repeat(np.arange(o.shape[1]).reshape((1, -1)), o.shape[0], axis=0), np.cast["int32"](y)]
    y_mask = y != -1
    cross_entropy = -np.mean(np.sum(np.log(p_hat) * y_mask, axis=1) / np.sum(y_mask, axis=1))
    return np.exp(cross_entropy)

gen_seeds = T.vector()
gen_len = T.lscalar()
[_, gen_seqs], gru_gen_updates = theano.scan(gru_step, outputs_info=[None, gen_seeds], n_steps=gen_len)
gen_seqs = T.cast(gen_seqs.dimshuffle((1, 0)), "int32")
gen_raw = theano.function([gen_seeds, gen_len], gen_seqs, updates=gru_gen_updates)

def generate(seedc, len_):
    seed = np.array([char_idxs[seedc]], dtype=floatX)
    preds = gen_raw(seed, len_ - 1)[0, :]
    return seedc + "".join(idx_chars[p] for p in preds)

def reset_gru_mem():
    for s in ss:
        s.set_value(np.zeros((args.batch_size, num_h), dtype=floatX), borrow=True)

def gen_sample():
    if not hasattr(gen_sample, "epochs"):
        gen_sample.epochs = 0
    else:
        gen_sample.epochs += 1
    with open(os.path.join(args.save_dir, "samples", "epoch{}".format(gen_sample.epochs)), "w") as f:
        print("Generating sample...", end="")
        print(generate(text_splits["train"][0], 10000), file=f)
        print("done")

θs = Wzs + Wrs + Whs + bzs + brs + bhs + [Wo, bo]
if args.optimizer == "adam":
    opt_fun = AdamAuto
elif args.optimizer == "dna":
    opt_fun = DnaAuto
elif args.optimizer == "rmsprop":
    opt_fun = RmspropAuto
optimizer = opt_fun(cross_entropy, θs, args.lr, dec=args.dec)
gen_sample()
batch_losses, epoch_losses, metric_results = fit_model(X, y, o, X_data, y_data, y_splits, cross_entropy, θs, optimizer, args.batch_size, args.num_epochs,
        save_dir=os.path.join(args.save_dir, "fitstats"), metrics={"perplexity": perplexity}, pre_train_clbks=[lambda: enable_dropout(train_premasks, train_postmasks)],
        patience=args.patience, sequential_train=True, post_train_clbks=[gen_sample, lambda: disable_dropout(train_premasks, train_postmasks)],
        eval_metrics_on=["train", "val", "test"], pre_pass_clbks=[reset_gru_mem], model_updates=gru_train_updates)

