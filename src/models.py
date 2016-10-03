from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2


def get_mlnn_model():
    model = Sequential()
    model.add(Dense(1000, activation="relu", W_regularizer=l2(0.0001), input_shape=(784, )))
    model.add(Dense(1000, activation="relu", W_regularizer=l2(0.0001)))
    model.add(Dense(10, activation="softmax", W_regularizer=l2(0.0001)))
    return model


MODEL_FACTORIES = {
    "mlnn": get_mlnn_model,
}

