from keras import layers
from keras.models import Model
import numpy as np


def model(num_days, num_neurons):
    temperature_input = layers.Input(shape=(num_days, 1))
    coordinate_input = layers.Input(shape=(2,))
    X = layers.LSTM(num_neurons, return_sequences=True)(temperature_input)
    coordinate_input = layers.RepeatVector(num_days)(coordinate_input)
    X = layers.Concatenate(axis=-1)(X, coordinate_input)
    X = layers.TimeDistributed(layers.Dense(1, activation='relu'))(X)
    rnn_model = Model(inputs=[temperature_input, coordinate_input], outputs=X)
    return rnn_model
