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
    def __init__(self, num_steps, embedding_size, seed=None, load_model=None):
        """Initializes the architecture of the LSTM and returns an instance.

        Args:
            num_steps:      An integer that is the number of unrolled steps that the LSTM takes. This is not (usually)
                            the length of the actual sequence. This number is the maximum
            embedding_size: An integer that is equal to the size of the vectors used to embed the input elements.
                            Example: 10,000 for 10,000 unique words in the vocabulary
            seed:           An integer used to seed the initial random state. Can be None to generate a new random seed.
            load_model:     If not None, then this should be a string indicating the checkpoint file containing data
                            that will be used to initialize the parameters of the model. Typically used when loading a
                            pre-trained model, or resuming a previous training session.
        """
        print("Constructing Architecture...")
        self._embedding_size = embedding_size
        self._seed = seed
        self._num_steps = num_steps  # Tuples are used to ensure the dimensions are immutable

        self._graph = tf.Graph()
        with self._graph.as_default():
            tf.set_random_seed(seed)
            batch_size = None  # Although this variable is never modified, it is present to enhance code readability

            with tf.variable_scope('Inputs'):
                self._x = tf.placeholder(tf.int32, shape=(batch_size, num_steps), name='X')
                self._y = tf.placeholder(tf.int32, shape=(batch_size,), name='Y')

                self._hot = tf.one_hot(indices=self._x, depth=embedding_size, name='Hot')  # X as one-hot encoded

            with tf.variable_scope('Unrolled') as scope:
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=256)  # This defines the cell structure
                state = lstm_cell.zero_state(batch_size=tf.shape(self._x)[0], dtype=tf.float32)  # Initial state

                for i in range(num_steps):
                    if i > 0: scope.reuse_variables()  # Reuse the variables created in the 1st LSTM cell
                    output, state = lstm_cell(self._hot[:, i, :], state)  # Step the LSTM through the sequence

            with tf.variable_scope('Softmax'):
                # Parameters
                w = tf.Variable(tf.random_normal((lstm_cell.output_size, embedding_size), stddev=0.1, name='Weights'))
                b = tf.Variable(tf.random_normal((embedding_size,), stddev=0.1, name='Bias'))
                scores = tf.matmul(output, w) + b  # The raw class scores to be fed into the loss function
                self._y_hat = tf.nn.softmax(scores, name='Y-Hat')  # Class probabilities, (batch_size, embedding_size)
                self._prediction = tf.argmax(self._y_hat, axis=1, name='Prediction')  # Vector of predicted classes

            with tf.variable_scope('Pipelining'):
                self._loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(  # Cross-entropy loss
                    logits=scores,
                    labels=self._y
                ), name='Loss')
                self._train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self._loss)

            self._sess = tf.Session()
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
