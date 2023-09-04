import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.models import Model


class DeepAE(object):
    def __init__(self, input_shape: int, layers_units: list[int]):
        self.input_shape = input_shape
        self.layers_units = layers_units
        self.Encoder = None
        self.Decoder = None
        self.Model = None
        self.Model_Encoder = None
        self.input_layer = Input(shape=(self.input_shape,))
        self.output_layer = Dense(units=self.input_shape, activation='sigmoid')

    def build_model(self):
        Dense_layers = self.__get_layers()

        # initialize the Encoder Model
        self.Encoder = Dense_layers[0](self.input_layer)
        for layer in Dense_layers[1:]:
            self.Encoder = layer(self.Encoder)

        # initialize the Decoder Model
        reversed_layers = self.__get_layers()
        reversed_layers = reversed_layers[::-1]
        reversed_layers = reversed_layers[1:]
        self.Decoder = reversed_layers[0](self.Encoder)
        for layer in reversed_layers[1:]:
            self.Decoder = layer(self.Decoder)
        self.Decoder = self.output_layer(self.Decoder)

        self.Model_Encoder = Model(inputs=self.input_layer, outputs=self.Encoder)
        self.Model = Model(inputs=self.input_layer, outputs=self.Decoder)

    def fit(self,
            x_train,
            optimizer_,
            loss,
            epochs,
            steps):

        if self.Model is None:
            return None

        # deep_autoencoder_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy')
        self.Model.compile(optimizer=optimizer_,
                           loss=loss)
        return self.Model.fit(x_train,
                              epochs=epochs,
                              steps_per_epoch=steps)

    def __get_layers(self):
        layers = []
        for num in self.layers_units:
            layers.append(Dense(units=num, activation='relu'))
        return layers



