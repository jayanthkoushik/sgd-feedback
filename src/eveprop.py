import keras.backend as K
from keras.optimizers import Optimizer


class Eveprop(Optimizer):

    def __init__(self, lr=0.001, rho=0.9, beta=0.999, epsilon=1e-8, decay=0.):
        super(Eveprop, self).__init__()
        self.__dict__.update(locals())
        self.iterations = K.variable(0)
        self.lr = K.variable(lr)
        self.rho = K.variable(rho)
        self.beta = K.variable(beta)
        self.decay = K.variable(decay)
        self.d = K.variable(1)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)

        self.updates = [K.update_add(self.iterations, 1)]
        t = self.iterations + 1

        loss_prev = K.variable(0)
        shapes = [K.get_variable_shape(p) for p in params]
        accumulators = [K.zeros(shape) for shape in shapes]

        ch_fact_lbound = K.switch(K.greater(loss, loss_prev), 1.1, 1/11.)
        ch_fact_ubound = K.switch(K.greater(loss, loss_prev), 11., 1/1.1)
        loss_ch_fact = loss / loss_prev
        loss_ch_fact = K.switch(K.lesser(loss_ch_fact, ch_fact_lbound), ch_fact_lbound, loss_ch_fact)
        loss_ch_fact = K.switch(K.greater(loss_ch_fact, ch_fact_ubound), ch_fact_ubound, loss_ch_fact)
        loss_hat = K.switch(K.greater(t, 1), loss_prev * loss_ch_fact, loss)

        d_den = K.switch(K.greater(loss_hat, loss_prev), loss_prev, loss_hat)
        d_t = (self.beta * self.d) + (1. - self.beta) * K.abs((loss_hat - loss_prev) / d_den)
        d_t = K.switch(K.greater(t, 1), d_t, 1.)
        self.updates.append(K.update(self.d, d_t))

        for p, g, a in zip(params, grads, accumulators):
            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            self.updates.append(K.update(a, new_a))
            new_p = p - (self.lr / (1. + (self.iterations * self.decay))) * g / ((K.sqrt(new_a) * d_t) + self.epsilon)
            self.updates.append(K.update(p, new_p))

        self.updates.append(K.update(loss_prev, loss_hat))
        return self.updates


    def get_config(self):
        config = {
            "lr": float(K.get_value(self.lr)),
            "rho": float(K.get_value(self.rho)),
            "beta": float(K.get_value(self.beta)),
            "decay": float(K.get_value(self.decay)),
            "epsilon": self.epsilon
        }
        base_config = super(Eveprop, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

