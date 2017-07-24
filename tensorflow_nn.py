# Reference - https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
import os
import warnings

warnings.filterwarnings("ignore")
from data import belgiumTS
import tensorflow as tf
import numpy as np
import functools


def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class Model:
    def __init__(self, image, label):
        self.image = image
        self.label = label
        self.prediction
        self.optimize
        self.error

    @lazy_property
    def prediction(self):
        # num_samples = int(self.image.shape[0])
        num_features = int(self.image.shape[1])
        num_classes = int(self.label.shape[1])
        num_hidden_nodes1 = 10

        w_layer1 = weight_variable(shape=[num_features, num_hidden_nodes1])
        b_layer1 = bias_variable(shape=[num_hidden_nodes1])

        # Layer 2 weights
        w_layer2 = weight_variable(shape=[num_hidden_nodes1, num_classes])
        b_layer2 = bias_variable(shape=[num_classes])

        input_layer = tf.matmul(self.image, w_layer1) + b_layer1
        hidden_layer = tf.nn.relu(input_layer)
        output_layer = tf.matmul(hidden_layer, w_layer2) + b_layer2

        return output_layer

    @lazy_property
    def optimize(self):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.prediction))
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        return optimizer.minimize(cross_entropy)

    @lazy_property
    def error(self):
        correct_predictions = tf.equal(tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        # Evaluate model
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return accuracy

def main():
    belgiumTS_data = belgiumTS.load_BelgiumTS()
    image = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 62])
    model = Model(image, label)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for _ in range(200):
        images, labels = belgiumTS_data.validation.images, belgiumTS_data.validation.labels
        acc = sess.run(model.error, {image: images, label: labels})
        print('Test accuracy {:6.2f}%'.format(100 * acc))
        for _ in range(61):
            images, labels = belgiumTS_data.train.next_batch(75)
            sess.run(model.optimize, {image: images, label: labels})
        print(belgiumTS_data.train.epochs_completed)
if __name__ == '__main__':
    main()