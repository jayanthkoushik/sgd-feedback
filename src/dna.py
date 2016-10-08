import keras.backend as K
from keras.optimizers import Optimizer
from keras.callbacks import Callback


class DNA(Optimizer):

    def __init__(self, lr, beta_1=0.9, beta_2=0.999, beta_3=0.999, epsilon=1e-8):
        super(DNA, self).__init__()
        self.__dict__.update(locals())
        self.iterations = K.variable(0)
        self.lr = K.variable(lr)
        self.beta_1 = K.variable(beta_1)
        self.beta_2 = K.variable(beta_2)
        self.beta_3 = K.variable(beta_3)
        self.d = K.variable(1)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)

        self.updates = [K.update_add(self.iterations, 1)]
        t = self.iterations + 1

        loss_prev = K.variable(0)
        shapes = [K.get_variable_shape(p) for p in params]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]

        d_t = (self.beta_3 * self.d) + (1. - self.beta_3) * K.abs((loss - loss_prev) / loss)
        self.updates.append(K.update(self.d, d_t))

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            mhat_t = m_t / (1. - K.pow(self.beta_1, t))
            self.updates.append(K.update(m, m_t))

            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            vhat_t = v_t / (1. - K.pow(self.beta_2, t))
            self.updates.append(K.update(v, v_t))

            p_t = p - self.lr * mhat_t / (K.sqrt(vhat_t) * d_t + self.epsilon)
            self.updates.append(K.update(p, p_t))

        self.updates.append(K.update(loss_prev, loss))
        return self.updates

    def get_config(self):
        config = {
	    "lr": float(K.get_value(self.lr)),
	    "beta_1": float(K.get_value(self.beta_1)),
	    "beta_2": float(K.get_value(self.beta_2)),
	    "epsilon": self.epsilon
        }
        base_config = super(DNA, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DNAMonitor(Callback):

    def on_train_begin(self, logs={}):
        self.batch_losses = []
        self.ds = []

    def on_batch_end(self, batch, logs={}):
        self.batch_losses.append(logs.get("loss"))
        self.ds.append(self.model.optimizer.d.get_value())

