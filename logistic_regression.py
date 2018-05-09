import tensorflow as tf
import numpy as np
from utils import define_scope
from random_generator import generate_classes_random_multifeature


class Model:
    # introduce batch size
    # introduce file size or somehow dataframe or something more generic
    # tf.summary
    def __init__(self, input, number_of_classes):
        self.input_size = input.shape[0]
        self.number_of_features = input.shape[1]
        self.number_of_classes = number_of_classes
        self.logits
        self.prediction
        self.optimizer
        self.loss
        self.error

    @define_scope
    def logits(self):
        weight = tf.Variable(tf.random_normal(shape=[self.number_of_features, self.number_of_classes]), name="weights")
        bias = tf.Variable(tf.random_normal(shape=[1, self.number_of_classes]), name="bias")
        return tf.matmul(self.input, weight) + bias

    @define_scope
    def loss(self):
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.logits)
        return tf.reduce_mean(entropy)

    @define_scope
    def prediction(self):
        return tf.nn.softmax(self.logits)

    @define_scope
    def optimizer(self):
        return tf.train.AdamOptimizer(0.01).minimize(self.loss)

    @define_scope
    def input(self):
        return tf.placeholder(tf.float32, [None, self.number_of_features])

    @define_scope
    def target(self):
        return tf.placeholder(tf.int32, [None, self.number_of_classes])

    @define_scope
    def error(self):
        mistakes = tf.not_equal(tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float64))


if __name__ == '__main__':
    number_of_classes = 3
    number_of_features = 3
    train_size = 1000
    valid_size = 100
    number_of_epochs = 1000
    x, y = generate_classes_random_multifeature(train_size, number_of_features, number_of_classes)
    valid_x, valid_y = generate_classes_random_multifeature(valid_size, number_of_features, number_of_classes)

    model = Model(x, number_of_classes)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(number_of_epochs):
            sess.run(model.optimizer, feed_dict={model.input: x, model.target: y})
            if i % 10 == 0:
                print(sess.run(model.loss, feed_dict={model.input: x, model.target: y}))
                print("error was ", sess.run(model.error, feed_dict={model.input: valid_x, model.target: valid_y}))
