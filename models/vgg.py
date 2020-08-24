#%%
import tensorflow as tf
from tensorflow import keras



class VGG16(keras.Model):
    def __init__(self, name='vgg16', dropout_rate=0.5):
        super(VGG16, self).__init__(name=name)
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
            'dropout1': keras.layers.Dropout(dropout_rate, name='dropout1'),
            'fc15': keras.layers.Dense(units=4096, activation='relu', kernel_regularizer=fc_regularizer, name='fc15'),
            'dropout2': keras.layers.Dropout(dropout_rate, name='dropout2'),
            'fc16': keras.layers.Dense(units=1000, activation='relu', kernel_regularizer=fc_regularizer, name='fc16'),
        }

    def call(self, x):
        for block in self.blocks:
            x = self.blocks[block](x)
        return x
    

class VGG19(keras.Model):
    def __init__(self, name='vgg19', dropout_rate=0.5):
        super(VGG19, self).__init__(name=name)
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
            'conv8': keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1,1), padding='same', name='conv8'),
            'maxpool3': keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), name='maxpool3'),

            'conv9': keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1,1), padding='same', name='conv9'),
            'conv10': keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1,1), padding='same', name='conv10'),
            'conv11': keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1,1), padding='same', name='conv11'),
            'conv12': keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1,1), padding='same', name='conv12'),
            'maxpool4': keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), name='maxpool4'),

            'conv13': keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1,1), padding='same', name='conv13'),
            'conv14': keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1,1), padding='same', name='conv14'),
            'conv15': keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1,1), padding='same', name='conv15'),
            'conv16': keras.layers.Conv2D(filters=512, kernel_size=3, strides=(1,1), padding='same', name='conv16'),
            'maxpool5': keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), name='maxpool5'),

            'flatten': keras.layers.Flatten(name='flatten'),
            'fc17': keras.layers.Dense(units=4096, activation='relu', kernel_regularizer=fc_regularizer, name='fc17'),
            'dropout1': keras.layers.Dropout(dropout_rate, name='dropout1'),
            'fc18': keras.layers.Dense(units=4096, activation='relu', kernel_regularizer=fc_regularizer, name='fc18'),
            'dropout2': keras.layers.Dropout(dropout_rate, name='dropout2'),
            'fc19': keras.layers.Dense(units=1000, activation='relu', kernel_regularizer=fc_regularizer, name='fc19'),
        }

    def call(self, x):
        for block in self.blocks:
            x = self.blocks[block](x)
        return x