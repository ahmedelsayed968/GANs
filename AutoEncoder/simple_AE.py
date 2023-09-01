import tensorflow as tf

from tensorflow.python.keras.layers import Dense
import numpy as np


def generate_data(m):
    """plots m random points on a 3D plane"""

    angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
    data = np.empty((m, 3))
    data[:, 0] = np.cos(angles) + np.sin(angles) / 2 + 0.1 * np.random.randn(m) / 2
    data[:, 1] = np.sin(angles) * 0.7 + 0.1 * np.random.randn(m) / 2
    data[:, 2] = data[:, 0] * 0.1 + data[:, 1] * 0.3 + 0.1 * np.random.randn(m)

    return data


class SimpleAE(object):
    def __init__(self):
        self.Encoder = None
        self.Decoder = None
        self.AutoEncoder = None

    def build_model(self):
        self.Encoder = tf.keras.Sequential([Dense(units=2,activation='relu', input_shape=[3])])
        self.Decoder = tf.keras.Sequential([Dense(units=3, input_shape=[2])])
        self.AutoEncoder = tf.keras.Sequential([self.Encoder, self.Decoder])

    def fit(self, x_train):
        if self.AutoEncoder is None:
            return
        self.AutoEncoder.compile(loss="mse",
                                 optimizer=tf.keras.optimizers.Adam())
        self.AutoEncoder.fit(x_train, x_train, epochs=50)


def test_model(model: SimpleAE, x_train):
    if model is None:
        return

    latent_space = model.Encoder.predict(x_train)
    reconstructed = model.Decoder.predict(latent_space)
    print(f'X-input: {x_train[0]}')
    print(f'Latent Space: {latent_space[0]}')
    print(f'Reconstruction: {reconstructed[0]}')
    """
        X-input: [0.62010126 0.52054213 0.11402505]
        Latent Space: [0.7358936 0.3219556]
        Reconstruction: [0.609344   0.4864387  0.22002548]
    """

if __name__ == '__main__':
    x_train = generate_data(100000)
    # normalize the data
    X_train = (x_train - np.mean(x_train, axis=0, keepdims=False))

    AE = SimpleAE()
    AE.build_model()
    AE.fit(X_train)
    test_model(AE, X_train)
