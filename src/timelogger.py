from time import time

from keras.callbacks import Callback


class TimeLogger(Callback):

    def on_train_begin(self, logs={}):
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.time_s = time()

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_times.append(time() - self.time_s)

