from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow import keras


class Resnet_block(keras.layers.Layer):
    def __init__(self, filter_a, filter_b, filter_c):
        super(Resnet_block, self).__init__()
        self.conv_a = tf.keras.layers.Conv2D(filter_a, (3, 3))
        self.batch_a = tf.keras.layers.BatchNormalization()
        self.conv_b = tf.keras.layers.Conv2D(filter_b, (3, 3))
        self.batch_b = tf.keras.layers.BatchNormalization()
        self.conv_c = tf.keras.layers.Conv2D(filter_c, (3, 3))
        self.batch_c = tf.keras.layers.BatchNormalization()



    def call(self, x, training):
        x = self.conv_a(x)
        x = self.batch_a(x, training)

        x = self.conv_b(x)
        x = self.batch_b(x, training)

        x = self.conv_c(x)
        x = self.batch_c(x, training)
        return x


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()

        self.resnet = Resnet_block(32, 64, 64)  
        self.flatten = Flatten()      
        self.d1 = Dense(128, activation='sigmoid')
        self.d2 = Dense(10)

    def call(self, x, training):
        x = self.resnet(x, training)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x