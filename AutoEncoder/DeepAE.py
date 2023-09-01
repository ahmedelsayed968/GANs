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
        self.input_layer = Input(shape=(self.input_shape,))
        self.output_layer = Dense(units=self.input_shape, activation='sigmoid')

    def build_model(self):
        Dense_layers = self.__get_layers()
        # initialize the Encoder Model
        self.Encoder = Dense_layers[0](self.input_layer)
        for layer in Dense_layers[1:]:
            self.Encoder = layer(self.Encoder)

        # initialize the Decoder Model
        reversed_layers = Dense_layers[::-1]
        self.Decoder = reversed_layers[0](self.Encoder)
        for layer in reversed_layers[1:]:
            self.Decoder = layer(self.Decoder)

        self.Model = Model(inputs=self.input_layer, outputs=self.Decoder)

    def fit(self,
            x_train,
            optimizer,
            loss,
            epochs):

        if self.Model is None:
            return None
        self.Model.compile(optimizer=optimizer,
                           loss=loss)
        self.Model.fit(x_train,
                       x_train,
                       epochs=epochs)

    def __get_layers(self):
        layers = []
        for num in self.layers_units:
            layers.append(Dense(units=num, activation='relu'))
        return layers
