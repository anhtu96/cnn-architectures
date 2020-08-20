#%%
import tensorflow as tf
from tensorflow import keras



class VGG16(keras.Model):
    def __init__(self):
        super(VGG16, self).__init__()
        fc_regularizer = keras.regularizers.L2(5e-4)
        self.blocks = {
            'conv1': keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), padding='same', name='conv1'),
            'conv2': keras.layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), padding='same', name='conv2'),
            'maxpool1': keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), name='maxpool1'),

            'conv3': keras.layers.Conv2D(filters=128, kernel_size=3, strides=(1,1), padding='same', name='conv3'),
            'conv4': keras.layers.Conv2D(filters=128, kernel_size=3, strides=(1,1), padding='same', name='conv4'),
            'maxpool2': keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), name='maxpool2'),
            
            'conv5': keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1,1), padding='same', name='conv5'),
            'conv6': keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1,1), padding='same', name='conv6'),
            'conv7': keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1,1), padding='same', name='conv7'),
            'maxpool3': keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), name='maxpool3'),

            'conv8': keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1,1), padding='same', name='conv8'),
            'conv9': keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1,1), padding='same', name='conv9'),
            'conv10': keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1,1), padding='same', name='conv10'),
            'maxpool4': keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), name='maxpool4'),

            'conv11': keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1,1), padding='same', name='conv11'),
            'conv12': keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1,1), padding='same', name='conv12'),
            'conv13': keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1,1), padding='same', name='conv13'),
            'maxpool5': keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), name='maxpool5'),

            'flatten': keras.layers.Flatten(name='flatten'),
            'fc14': keras.layers.Dense(units=4096, activation='relu', kernel_regularizer=fc_regularizer, name='fc14'),
            'dropout1': keras.layers.Dropout(0.5, name='dropout1'),
            'fc15': keras.layers.Dense(units=4096, activation='relu', kernel_regularizer=fc_regularizer, name='fc15'),
            'dropout2': keras.layers.Dropout(0.5, name='dropout2'),
            'fc16': keras.layers.Dense(units=1000, activation='relu', kernel_regularizer=fc_regularizer, name='fc16'),
        }

    def call(self, x):
        for block in self.blocks:
            x = self.blocks[block](x)
        return x