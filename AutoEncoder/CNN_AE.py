import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Input
from tensorflow.python.keras.models import Model


class CNN_AE(object):
    def __init__(self, layers_filters, input_shape):
        self.layers_filters = layers_filters
        self.input_shape = input_shape
        self.Encoder = None
        self.Model = None
        self.Decoder = None
        self.Model_Encoder = None
        self.input_layer = Input(shape=input_shape)
        self.history = None

    def build_model(self):
        self.__build_encoder()
        self.__build_decoder()

    def __build_encoder(self):
        cnn_layers = self.__get_layers()
        self.Encoder = cnn_layers[0](self.input_layer)
        self.Encoder = MaxPool2D(pool_size=(2, 2))(self.Encoder)
        for layer in cnn_layers[1:-1]:
            self.Encoder = layer(self.Encoder)
            self.Encoder = MaxPool2D(pool_size=(2, 2))(self.Encoder)
        self.Encoder = cnn_layers[-1](self.Encoder)
        encoder_visualization = Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')(
            self.Encoder)
        self.Model_Encoder = Model(inputs=self.input_layer, outputs=encoder_visualization)

    def __build_decoder(self):
        if self.Encoder is None:
            return None
        cnn_layers = self.__get_layers()
        cnn_layers = (cnn_layers[::-1])
        cnn_layers = cnn_layers[1:]

        self.Decoder = cnn_layers[0](self.Encoder)
        self.Decoder = UpSampling2D(size=(2, 2))(self.Decoder)
        for layer in cnn_layers[1:]:
            self.Decoder = layer(self.Decoder)
            self.Decoder = UpSampling2D(size=(2, 2))(self.Decoder)
        self.Decoder = Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')(self.Decoder)
        self.Model = Model(inputs=self.input_layer, outputs=self.Decoder)

    def fit(self,
            train,
            valid_ds,
            optimizer,
            loss,
            train_steps,
            valid_steps,
            epochs):
        try:
            self.Model.compile(loss=loss, optimizer=optimizer)
            self.history = self.Model.fit(train,
                                          steps_per_epoch=train_steps,
                                          validation_data=valid_ds,
                                          validation_steps=valid_steps,
                                          epochs=epochs)
        except Exception as E:
            print(E)

    def __get_layers(self):
        layers = []
        for i in self.layers_filters:
            layers.append(Conv2D(filters=i, kernel_size=(3, 3), padding='same', activation='relu'))
        return layers



