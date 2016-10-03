from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.regularizers import l2


def get_mlnn_model(input_shape, nb_classes):
    model = Sequential()
    model.add(Dense(1000, activation="relu", W_regularizer=l2(0.0001), input_shape=input_shape))
    model.add(Dense(1000, activation="relu", W_regularizer=l2(0.0001)))
    model.add(Dense(nb_classes, activation="softmax", W_regularizer=l2(0.0001)))
    return model


def get_cnn_model(input_shape, nb_classes):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, activation="relu", border_mode="same", input_shape=input_shape))
    model.add(Convolution2D(32, 3, 3, activation="relu", border_mode="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, activation="relu", border_mode="same"))
    model.add(Convolution2D(64, 3, 3, activation="relu", border_mode="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation="softmax"))
    return model


MODEL_FACTORIES = {
    "mlnn": get_mlnn_model,
    "cnn": get_cnn_model
}

