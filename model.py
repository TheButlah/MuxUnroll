from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


'''
#from __future__ import unicode_literals

# noinspection PyCompatibility
from builtins import range
from future import standard_library
standard_library.install_aliases()'''

import tensorflow as tf
import numpy as np

from time import time, strftime
import os


class LSTM(object):

    def __init__(self, input_shape, embedding_size, seed=None, load_model=None):
        """Initializes the architecture of the LSTM and returns an instance.

        Args:
            input_shape:    A list that represents the shape of the input. Can contain None as the first element to
                            indicate that the batch size can vary (this is the preferred way to do it). Second element
                            should be the number of unrolled steps that the LSTM takes. Example: [None, 20]
            embedding_size: An integer that is equal to the size of the vectors used to embed the input elements.
                            Example: 10,000 for 10,000 unique words in the vocabulary
            seed:           An integer used to seed the initial random state. Can be None to generate a new random seed.
            load_model:     If not None, then this should be a string indicating the checkpoint file containing data
                            that will be used to initialize the parameters of the model. Typically used when loading a
                            pre-trained model, or resuming a previous training session.
        """
        print("Constructing Architecture...")
        self._input_shape = tuple(input_shape)  # Tuples are used to ensure the dimensions are immutable
        x_shape = tuple(input_shape)  # 1st dim should be the size of dataset

        ####DONE UP TO HERE I THINK###

        y_shape = tuple(input_shape[:-1])  # Rank of y should be one less
        self._embedding_size = embedding_size
        self._seed = seed
        self._graph = tf.Graph()
        with self._graph.as_default():

            with tf.variable_scope('Pipelining'):
                self._loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=self._y),
                    name='Loss'
                )
                self._train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self._loss)

            self._sess = tf.Session(graph=self._graph)  # Not sure if this really needs to explicitly specify the graph
            with self._sess.as_default():
                self._saver = tf.train.Saver()
                if load_model is not None:
                    print("Restoring Model...")
                    load_model = os.path.abspath(load_model)
                    self._saver.restore(self._sess, load_model)
                    print("Model Restored!")
                else:
                    print("Initializing model...")
                    self._sess.run(tf.global_variables_initializer())
                    print("Model Initialized!")

    def train(self, x_train, y_train, num_epochs, start_stop_info=True, progress_info=True):
        with self._sess.as_default():
            # Training loop for parameter tuning
            if start_stop_info:
                print("Starting training for %d epochs" % num_epochs)
            last_time = time()
            for epoch in range(num_epochs):
                _, loss_val = self._sess.run(
                    [self._train_step, self._loss],
                    feed_dict={self._x: x_train, self._y: y_train, self._phase_train: True}
                )
                current_time = time()
                if progress_info and (current_time - last_time) >= 5:  # Only print progress every 5 seconds
                    last_time = current_time
                    print("Current Loss Value: %.10f, Percent Complete: %.4f" % (loss_val, epoch / num_epochs * 100))
            if start_stop_info:
                print("Completed Training.")
            return loss_val

    def apply(self, x_data):
        """Applies the model to the batch of data provided. Typically called after the model is trained.

        Args:
            x_data:  A numpy ndarray of the data to apply the model to. Should have the same shape as the training data.

        Returns:
            A numpy ndarray of the data, with the last dimension being the class probabilities instead of channels.
            Example: result.shape is [batch_size, 640, 480, 10] for a 640x480 RGB image with 10 target classes
        """
        with self._sess.as_default():
            return self._sess.run(self._y_hat, feed_dict={self._x: x_data, self._phase_train: False})

    def save_model(self, save_path=None):
        """Saves the model in the specified file.

        Args:
            save_path:  The relative path to the file. By default, it is
                saved/GenSeg-Year-Month-Date_Hour-Minute-Second.ckpt
        """
        with self._sess.as_default():
            print("Saving Model")
            if save_path is None:
                save_path = "saved/GenSeg-%s.ckpt" % strftime("%Y-%m-%d_%H-%M-%S")
            dirname = os.path.dirname(save_path)
            if dirname is not '':
                os.makedirs(dirname, exist_ok=True)
            save_path = os.path.abspath(save_path)
            path = self._saver.save(self._sess, save_path)
            print("Model successfully saved in file: %s" % path)

