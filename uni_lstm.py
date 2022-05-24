from keras import layers
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt


class rnn_model:
    def __init__(self, num_days, num_neurons_lstm, num_a_densor):
        self.numDays = num_days
        temperature_input = layers.Input(shape=(num_days, 1))
        coordinate_input = layers.Input(shape=(2,))
        X = layers.LSTM(num_neurons_lstm, return_sequences=True)(temperature_input)
        repeated_coordinate_input = layers.RepeatVector(num_days)(coordinate_input)
        X = layers.Concatenate(axis=-1)([X, repeated_coordinate_input])
        X = layers.TimeDistributed(layers.Dense(num_a_densor, activation='tanh'))(X)
        X = layers.TimeDistributed(layers.Dense(1))(X)
        self.model = Model(inputs=[temperature_input, coordinate_input], outputs=X)

    def train(self, X, Y, epochs, batch_size):
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['MeanSquaredError'])
        self.model.fit(X, Y, epochs=epochs, batch_size=batch_size)

    def test(self, X, Y):
        newX = X
        for i in range(self.numDays):
            print(f'Predicting {i}th date...')
            prediction = self.model.predict(newX)[:, [-1], :]
            newX = [np.concatenate((newX[0][:, 1:, :], prediction), axis=1), newX[1]]
        trueX = newX[0] * 10 + 60
        trueY = Y * 10 + 60
        plt.plot(trueX[0, :, 0], label='prediction')
        plt.plot(trueY[0, :, 0], label='GT')
        plt.legend()
        plt.show()
        return np.mean(np.square((newX[0] * 10 + 60) - (Y * 10 + 60)))


if __name__ == '__main__':
    X_temp_train = np.load('data/X_temp_train.npy')
    X_coordinate_train = np.load('data/X_coordinate_train.npy')
    Y_temp_train = np.load('data/Y_temp_train.npy')
    X_temp_test = np.load('data/X_temp_test.npy')
    X_coordinate_test = np.load('data/X_coordinate_test.npy')
    Y_temp_test = np.load('data/Y_temp_test.npy')
    model = rnn_model(365, 10, 5)
    model.train([X_temp_train, X_coordinate_train], Y_temp_train, 15, 128)
    print(model.test([X_temp_test, X_coordinate_test], Y_temp_test))

