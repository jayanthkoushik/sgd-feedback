import pickle
import os
import math
import functools
from collections import defaultdict

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from tqdm import tqdm
from tabulate import tabulate
from sklearn.externals import joblib


print = functools.partial(print, flush=True)
floatX = theano.config.floatX
one = np.cast[floatX](1)


def free_shared_variable(x):
    x.set_value(np.empty([1 for _ in range(x.ndim - 1)] + [0], dtype=x.dtype))


class RmspropAuto:

    typ = "auto"

    def __init__(self, f, θs, α=0.001, ρ=0.9, ε=1e-8, dec=0.):
        α, ρ, ε, dec = [np.cast[floatX](h) for h in [α, ρ, ε, dec]]

        t = theano.shared(0, name="t")
        t_u = (t, t + 1)

        gs = T.grad(f, θs)

        self.accs = [theano.shared(np.zeros(θ.shape.eval(), dtype=floatX), borrow=True, name="a") for θ in θs]
        acc_us = [(acc, (ρ * acc) + (one - ρ) * T.sqr(g)) for acc, g in zip(self.accs, gs)]

        θ_us = [(θ, θ - ((α / (one + (t_u[1] * dec))) * g / T.sqrt(acc_u[1] + ε))) for θ, g, acc_u in zip(θs, gs, acc_us)]
        self.updates = acc_us + θ_us + [t_u]

    def __call__(self):
        return self.updates

    def free_shared(self):
        for p in self.accs:
            free_shared_variable(p)


class EveAuto:

    typ = "auto"

    def __init__(self, f, θs, α=0.001, β1=0.9, β2=0.999, β3=0.999, k=0.1, K=10., ε=1e-8, dec=0.):
        α, β1, β2, β3, ε, dec = [np.cast[floatX](h) for h in [α, β1, β2, β3, ε, dec]]

        t = theano.shared(0, name="t")
        t_u = (t, t + 1)

        f_prev = theano.shared(np.cast[floatX](0), name="f_prev")

        ch_fact_lbound = T.switch(T.gt(f, f_prev), 1+k, 1/(1+K))
        ch_fact_ubound = T.switch(T.gt(f, f_prev), 1+K, 1/(1+k))
        f_ch_fact = f / f_prev
        f_ch_fact = T.switch(T.lt(f_ch_fact, ch_fact_lbound), ch_fact_lbound, f_ch_fact)
        f_ch_fact = T.switch(T.gt(f_ch_fact, ch_fact_ubound), ch_fact_ubound, f_ch_fact)
        f_hat = T.switch(T.gt(t_u[1], 1), f_prev * f_ch_fact, f)
        f_u = (f_prev, f_hat)

        self.ms = [theano.shared(np.zeros(θ.shape.eval(), dtype=floatX), borrow=True, name="m") for θ in θs]
        self.vs = [theano.shared(np.zeros(θ.shape.eval(), dtype=floatX), borrow=True, name="v") for θ in θs]

        d = theano.shared(one, name="d")
        d_den = T.switch(T.gt(f_hat, f_prev), f_prev, f_hat)
        d_t = (β3 * d) + (one - β3) * T.abs_((f_hat - f_prev) / d_den)
        d_t = T.switch(T.gt(t_u[1], one), d_t, one)
        d_u = (d, d_t)

        gs = T.grad(f, θs)

        m_us = [(m, β1 * m + (one - β1) * g) for m, g in zip(self.ms, gs)]
        m_hats = [m_u[1] / (one - T.pow(β1, t_u[1])) for m_u in m_us]

        v_us = [(v, β2 * v + (one - β2) * T.sqr(g)) for v, g in zip(self.vs, gs)]
        v_hats = [v_u[1] / (one - T.pow(β2, t_u[1])) for v_u in v_us]

        θ_us = [(θ, θ - (α / (one + (t_u[1] * dec))) * m_hat / ((T.sqrt(v_hat) * d_t) + ε)) for θ, m_hat, v_hat in zip(θs, m_hats, v_hats)]
        self.updates = m_us + v_us + [t_u, f_u, d_u] + θ_us

    def __call__(self):
        return self.updates

    def free_shared(self):
        for p in self.ms + self.vs:
            free_shared_variable(p)


class AdamAuto:

    typ = "auto"

    def __init__(self, f, θs, α=0.001, β1=0.9, β2=0.999, ε=1e-8, dec=0.):
        """
        f: tensor representing the loss function.
        θs: list of shared variables representing the parameters.
        The remaining parameters are the same as in the paper.
        """
        α, β1, β2, ε, dec = [np.cast[floatX](h) for h in [α, β1, β2, ε, dec]]

        t = theano.shared(0, name="t")
        self.ms = [theano.shared(np.zeros(θ.shape.eval(), dtype=floatX), borrow=True, name="m") for θ in θs]
        self.vs = [theano.shared(np.zeros(θ.shape.eval(), dtype=floatX), borrow=True, name="v") for θ in θs]

        gs = T.grad(f, θs)
        t_u = (t, t + 1)
        m_us = [(m, β1 * m + (one - β1) * g) for m, g in zip(self.ms, gs)]
        v_us = [(v, β2 * v + (one - β2) * T.sqr(g)) for v, g in zip(self.vs, gs)]
        α_hat = α * T.sqrt(one - T.cast(T.pow(β2, t_u[1]), floatX)) / (one - T.cast(T.pow(β1, t_u[1]), floatX))
        α_hat = α_hat / (one + (t_u[1] * dec))
        θ_us = [(θ, θ - α_hat * m_u[1] / (T.sqrt(v_u[1]) + ε)) for θ, m_u, v_u in zip(θs, m_us, v_us)]
        self.updates = m_us + v_us + [t_u] + θ_us

    def __call__(self):
        return self.updates

    def free_shared(self):
        for p in self.ms + self.vs:
            free_shared_variable(p)


class AdamManual:

    typ = "manual"

    def __init__(self, θs, α=0.001, β1=0.9, β2=0.999, ε=1e-8):
        """
        θs: list of shared variables representing the parameters.
        The remaining parameters are the same as in the paper.
        """
        self.θs = θs
        self.α, self.β1, self.β2, self.ε = [np.cast[floatX](h) for h in [α, β1, β2, ε]]
        self.t = 0
        self.ms = [np.zeros(θ.shape.eval(), dtype=floatX) for θ in θs]
        self.vs = [np.zeros(θ.shape.eval(), dtype=floatX) for θ in θs]

    def __call__(self, gs):
        """
        gs: list of gradients with respect to each of the parameters.
        """
        self.t += 1
        for i, g in enumerate(gs):
            self.ms[i] = self.β1 * self.ms[i] + (one - self.β1) * g
            self.vs[i] = self.β2 * self.vs[i] + (one - self.β2) * np.square(g)
        α_hat = self.α * np.sqrt(one - np.cast[floatX](np.power(self.β2, self.t))) / (one - np.cast[floatX](np.power(self.β1, self.t)))
        for θ, m, v in zip(self.θs, self.ms, self.vs):
            θ_old = θ.get_value(borrow=True, return_intern_type=True)
            θ.set_value(θ_old - α_hat * m / (np.sqrt(v) + self.ε), borrow=True)


class AdamaxAuto:

    typ = "auto"

    def __init__(self, f, θs, α=0.002, β1=0.9, β2=0.999, ε=1e-8, dec=0.):
        α, β1, β2, ε, dec = [np.cast[floatX](h) for h in [α, β1, β2, ε, dec]]

        t = theano.shared(0, name="t")
        self.ms = [theano.shared(np.zeros(θ.shape.eval(), dtype=floatX), borrow=True, name="m") for θ in θs]
        self.us = [theano.shared(np.zeros(θ.shape.eval(), dtype=floatX), borrow=True, name="u") for θ in θs]

        gs = T.grad(f, θs)
        t_u = (t, t + 1)
        m_us = [(m, β1 * m + (one - β1) * g) for m, g in zip(self.ms, gs)]
        u_us = [(u, T.cast(T.maximum(β2 * u, T.abs_(g)), floatX)) for u, g in zip(self.us, gs)]
        α_hat = α / (one - T.cast(T.pow(β1, t_u[1]), floatX))
        α_hat = α_hat / (one + (t_u[1] * dec))
        θ_us = [(θ, θ - α_hat * m_u[1] / (u_u[1] + ε)) for θ, m_u, u_u in zip(θs, m_us, u_us)]
        self.updates = m_us + u_us + [t_u] + θ_us

    def __call__(self):
        return self.updates

    def free_shared(self):
        for p in self.ms + self.vs:
            free_shared_variable(p)


def dropout_layer(input_, p, srng, train_premasks, train_postmasks, dropout_mask=None):
    """
    input_: tensor on which dropout should be applied.
    p: the dropout probability.
    srng: theano shared random stream.
    train_premasks: a list of masks that control whether or not units are dropped.
    train_postmasks: a list of masks that scale output by 0.5 after training.
    dropout_mask: pre-made dropout mask.
    """
    train_premask = theano.shared(1)
    train_postmask = theano.shared(np.cast[floatX](1))
    train_premasks.append(train_premask)
    train_postmasks.append(train_postmask)
    if dropout_mask is None:
        dropout_mask = srng.binomial(size=input_.shape, p=p) * train_premask
    input_drop = T.switch(dropout_mask, 0, input_) * train_postmask
    return input_drop


def enable_dropout(train_premasks, train_postmasks):
    """The arguments have the same meaning as in the dropout_layer function."""
    for premask in train_premasks:
        premask.set_value(1)
    for postmask in train_postmasks:
        postmask.set_value(np.cast[floatX](1))


def disable_dropout(train_premasks, train_postmasks):
    """The arguments have the same meaning as in the dropout_layer function."""
    for premask in train_premasks:
        premask.set_value(0)
    for postmask in train_postmasks:
        postmask.set_value(np.cast[floatX](0.5))


def generate_batch_idxs(num_samples, batch_size, mode):
    """
    num_samples: total number of samples in 1 epoch.
    batch_size: number of samples in 1 minibatch.
    mode: "sequential" / "random" - order for generating batch indexes.
    """
    if mode == "sequential":
        idxs = range(num_samples)
    else:
        idxs = np.random.choice(num_samples, size=num_samples, replace=False)
    return [idxs[i:i + batch_size] for i in range(0, num_samples, batch_size)]


def eval_model(X, y_hat, X_data, y_splits, metrics, eval_on, batch_size, save_dir=None, pre_pass_clbks=[], model_updates=[], plot_avp=False):
    """
    X / y_hat: tensor representing the model inputs / predictions.
    X_data: dictionary mapping eval_on to corresponding theano shared variables with feature arrays.
    y_splits: dictionary mapping eval_on to corresponding numpy label arrays.
    metrics: dictionary mapping metric names to scalar functions of the form metric(y, y_hat).
    eval_on: sublist of ["train", "val", "test"] - data splits on which to evaluate.
    batch_size: number of samples per minibatch.
    save_dir: directory to save actual vs. predicted curves.
    pre_pass_clbks: functions to call before each pass through a data split.
    model_updates: list of model specific updates to be added to theano functions.
    plot_avp: whether to plot actual vs predicted curves.
    """
    if save_dir is not None and plot_avp is True and not os.path.exists(os.path.join(save_dir, "avp")):
        os.makedirs(os.path.join(save_dir, "avp"))

    eval_on = list(eval_on)
    batch_idxs = T.lvector()
    y_hat_splits = defaultdict(list)
    for split in eval_on:
        for pre_pass_clbk in pre_pass_clbks:
            pre_pass_clbk()

        pred_function = theano.function([batch_idxs], y_hat, givens={X: X_data[split][batch_idxs, :]}, updates=model_updates)
        batch_idxs_values = generate_batch_idxs(X_data[split].shape.eval()[0], batch_size, "sequential")
        for batch_idxs_value in tqdm(batch_idxs_values, desc="Predicting on {} set".format(split), ncols=100, ascii=False, unit="batch"):
            y_hat_splits[split].extend(pred_function(batch_idxs_value))
        y_hat_splits[split] = np.array(y_hat_splits[split])
        pred_function.free()

        if plot_avp:
            fig = plt.figure(figsize=(8, 6))
            plt.scatter(y_splits[split], y_hat_splits[split], label="Actual vs predicted")
            plt.plot(y_splits[split], y_splits[split], label="Actual vs actual")
            plt.xlabel("Actual labels")
            plt.ylabel("Predicted labels")
            plt.legend(loc="upper left")
            if save_dir is not None:
                fig.savefig(os.path.join(save_dir, "avp", "{}.png".format(split)), dpi=500)
            else:
                plt.show(block=False)
            plt.close(fig)

    results = []
    for metric_name, metric in metrics.items():
        results.append([metric_name])
        for split in eval_on:
            results[-1].append(metric(y_splits[split], y_hat_splits[split]))

    print()
    print(tabulate(results, headers=["Metric"] + [s.title() for s in eval_on], numalign="left", stralign="left"))
    return results


def fit_model(X, y, y_hat, X_data, y_data, y_splits, f, θs, optimizer, batch_size, epochs, patience=np.inf, min_epochs=0, save_dir=None, metrics=None,
              eval_metrics_on=["train", "val"], pre_train_clbks=[], post_train_clbks=[], pre_pass_clbks=[], model_updates=[], sequential_train=False, plot_avp=False):
    """
    X / y / y_hat: tensor representing the model inputs / labels / predictions.
    X_data / y_data: dictionary mapping "train", "val", ("test") to corresponding theano shared variables with feature / label arrays.
    y_splits: dictionary mapping eval_metrics_on to corresponding numpy label arrays.
    f: tensor representing the loss function.
    θs: list of tensors representing the parameters.
    optimizer: an optimizer callable; arguments depend on the type ("auto" / "manual").
    batch_size: number of samples per minibatch.
    epochs: number of training epochs.
    patience: early stopping parameter; training is stopped if validation loss does not decrease for this many consecutive epochs.
    min_epochs: early stopping logic begins after this many epochs.
    save_dir: directory used for saving model parameters and loss curves.
    metrics: metrics to compute after training.
    eval_metrics_on: sublist of ["train", "val", "test"] on which to evaluate given metrics.
    pre_train_clbks: functions to call before the training phase of each epoch.
    post_train_clbks: functions to call after the training phase of each epoch.
    pre_pass_clbks: functions to call before each pass through a data split.
    model_updates: list of model specific updates to be added to theano functions.
    sequential_train: whether to make training batches sequential.
    """
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(os.path.join(save_dir, "model")):
            os.makedirs(os.path.join(save_dir, "model"))

    print("Compiling functions...", end="")
    batch_idxs = T.lvector()
    givens = {X: X_data["train"][batch_idxs, :]}
    if y is not None:
        givens[y] = y_data["train"][batch_idxs, :]
    if optimizer.typ == "auto":
        train = theano.function([batch_idxs], f, updates=optimizer() + model_updates, givens=givens)
    else:
        train = theano.function([batch_idxs], [f, *T.grad(f, θs)], updates=model_updates, givens=givens)

    split_sizes = {}
    for split in ["train", "val"]:
        split_sizes[split] = X_data[split].shape.eval()[0]

    evaluates = {}
    for split in ["train", "val"]:
        givens = {X: X_data[split][batch_idxs, :]}
        if y is not None:
            givens[y] = y_data[split][batch_idxs, :]
        evaluates[split] = theano.function([batch_idxs], f, updates=model_updates, givens=givens)
    print("done\n")

    tr_batch_idxs = generate_batch_idxs(split_sizes["train"], batch_size, "sequential" if sequential_train else "random")
    seq_batch_idxs = {}
    for split in ["train", "val"]:
        seq_batch_idxs[split] = generate_batch_idxs(split_sizes[split], batch_size, "sequential")

    batch_losses = []
    epoch_losses = {"train": [], "val": []}
    best_val_loss = np.inf
    best_θ_values = None
    impatience = 0
    for epoch in range(1, epochs + 1):
        for pre_train_clbk in pre_train_clbks:
            pre_train_clbk()

        for pre_pass_clbk in pre_pass_clbks:
            pre_pass_clbk()

        for batch_idxs_value in tqdm(tr_batch_idxs, desc="Epoch {} training".format(epoch), ncols=100, ascii=False, unit="batch"):
            if optimizer.typ == "auto":
                batch_losses.append(train(batch_idxs_value).item())
            else:
                train_ret = train(batch_idxs_value)
                loss, gs = train_ret[0], [np.array(ng) for ng in train_ret[1:]]
                batch_losses.append(loss.item())
                optimizer(gs)

        for post_train_clbk in post_train_clbks:
            post_train_clbk()

        for split in ["train", "val"]:
            for pre_pass_clbk in pre_pass_clbks:
                pre_pass_clbk()

            loss = 0.
            for batch_idxs_value in tqdm(seq_batch_idxs[split], desc="Evaluating loss on {} set".format(split), ncols=100, ascii=False, unit="batch"):
                loss += evaluates[split](batch_idxs_value) * len(batch_idxs_value)
            epoch_losses[split].append(loss / split_sizes[split])
        print("Loss: {} {}, {} {}".format("train", epoch_losses["train"][-1], "val", epoch_losses["val"][-1]))

        epoch_val_loss = epoch_losses["val"][-1]
        if epoch > min_epochs:
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_θ_values = [θ.get_value() for θ in θs]
                impatience = 0
            else:
                impatience += 1
                if impatience == patience:
                    print("Stopping early at {} epochs".format(epoch))
                    break
        print()

    train.free()
    for split in ["train", "val"]:
        evaluates[split].free()

    if best_θ_values is not None:
        for θ, best_θ_value in zip(θs, best_θ_values):
            θ.set_value(best_θ_value, borrow=True)
    else:
        best_θ_values = [θ.get_value(borrow=True) for θ in θs]
    if save_dir is not None:
        joblib.dump(best_θ_values, os.path.join(save_dir, "model", "model.pkl"))

    fig = plt.figure(figsize=(16, 6))
    for i, split in zip([1, 2], ["train", "val"]):
        plt.subplot(1, 2, i)
        plt.plot(epoch_losses[split])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("{} losses".format(split.title()))

    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, "loss_curves.png"), dpi=500)
        with open(os.path.join(save_dir, "batch_losses.pkl"), "wb") as f:
            pickle.dump(batch_losses, f)
        with open(os.path.join(save_dir, "epoch_losses.pkl"), "wb") as f:
            pickle.dump(epoch_losses, f)
    else:
        plt.show(block=False)
    plt.close(fig)

    if metrics is not None:
        metric_results = eval_model(X, y_hat, X_data, y_splits, metrics, eval_metrics_on, batch_size, save_dir, pre_pass_clbks, model_updates, plot_avp)
        if save_dir is not None:
            with open(os.path.join(save_dir, "metrics.pkl"), "wb") as f:
                pickle.dump(metric_results, f)
        return batch_losses, epoch_losses, metric_results
    else:
        return batch_losses, epoch_losses

