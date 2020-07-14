from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow import keras






class Multiply(keras.layers.Layer):
    def __init__(self):
        super(Multiply, self).__init__()
        self.c = tf.Variable(1, trainable=True, dtype='float32')

    def call(self, x):
        print(self.c)
        return self.c * x




class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.mu = Multiply()


        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.mu(x)
        x = self.d1(x)
        return self.d2(x)


#     def sum_of_weights(self):
#         #weights = tf.Variable(0, trainable=False, dtype = "float32")
#         weights = 0.0
#         for layer in self.layers:
#             for params in layer.trainable_weights:
#                 weights = tf.add(weights, tf.reduce_sum(tf.pow(params, 2)))
#         return weights