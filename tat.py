import tensorflow as tf
import numpy as np
from Model import MyModel

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# Add a channels dimension



# x_train = np.reshape(x_train, [x_train.shape[0], x_train.shape[1] * x_train.shape[2]])
# x_test = np.reshape(x_test, [x_test.shape[0], x_test.shape[1] * x_test.shape[2]])



x_train = x_train[..., np.newaxis].astype("float32")
x_test = x_test[..., np.newaxis].astype("float32")






train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)



# Create an instance of the model
model = MyModel()


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()



@tf.function
def train_step(images, labels):
    #print(images.shape)
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions) #+ model.sum_of_weights()
    gradients = tape.gradient(loss, model.trainable_variables)
    # change model parameters
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return predictions


#@tf.function
def test_step(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    return [np.argmax(p) for p in predictions]



EPOCHS = 20

for epoch in range(EPOCHS):

    predictions = []
    inpts = []
    for images, labels in train_ds:
        tmp = train_step(images, labels)
        predictions.extend([np.argmax(i) for i in tmp])
        inpts.extend(labels)

    train_acc = sum([x == y.numpy() for x,y in zip(predictions, inpts)]) / len(predictions)

    predictions = []
    for test_images, test_labels in test_ds:
        predictions.extend(test_step(test_images, test_labels))


    test_acc = sum([1 for x,y in zip(y_test, predictions) if x == y]) / len(y_test)

    print('Epoch: {}, train_acc: {:1f}, test_acc: {:1f}'.format(epoch + 1, train_acc, test_acc))


    # template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    # print(template.format(epoch + 1,
    #                         train_loss.result(),
    #                         train_accuracy.result() * 100,
    #                         test_loss.result(),
    #                         test_accuracy.result() * 100))