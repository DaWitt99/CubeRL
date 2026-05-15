import tensorflow as tf

def make_model(model_type):
    if model_type == "Dense_4":
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(324,)),
            DenseBN(units=4096, activation='elu'),
            DenseBN(units=2048, activation='elu'),
            DenseBN(units=2048, activation='elu'),
            DenseBN(units=1024, activation='elu'),
            tf.keras.layers.Dense(18),
        ])
    elif model_type == "Residual_4":
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(324,)),
            DenseBN(units=4096, activation='elu'),
            ResidualBlock(4096,"elu"),
            ResidualBlock(2048,"elu"),
            ResidualBlock(2048,"elu"),
            ResidualBlock(1024,"elu"),
            tf.keras.layers.Dense(18)
        ])
    else:
        raise Exception("Invalid type")

    return model

class DenseBN(tf.keras.layers.Layer):
    def __init__(self,units=64, activation='elu', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.dense = tf.keras.layers.Dense(units=self.units, kernel_initializer="he_normal")
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.bn(x)
        x= self.activation(x)
        return x

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'units': self.units, 'activation': tf.keras.activations.serialize(self.activation)}

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, units = 128, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.layers = [
            tf.keras.layers.Dense(units=self.units, kernel_initializer="he_normal"),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            tf.keras.layers.Dense(units=self.units, kernel_initializer="he_normal"),
            tf.keras.layers.BatchNormalization(),
        ]
        self.skip_layers = [
            tf.keras.layers.Dense(units=self.units, kernel_initializer="he_normal"),
            tf.keras.layers.BatchNormalization(),
        ]

    def call(self, inputs):
        Y = inputs
        for layer in self.layers:
            Y=layer(Y)

        Y_skip = inputs
        for skip_layer in self.skip_layers:
            Y_skip = skip_layer(Y_skip)

        return self.activation(Y+Y_skip)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'units': self.units, 'activation': tf.keras.activations.serialize(self.activation)}