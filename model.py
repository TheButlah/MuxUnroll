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
                            the length of the actual sequence
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
            batch_size = None

            with tf.variable_scope('Inputs'):
                self._x = tf.placeholder(tf.int32, shape=(batch_size, num_steps), name='X')
                self._hot = tf.one_hot(indices=self._x, depth=embedding_size, name='Hot')
                self._y = tf.placeholder(tf.int32, shape=(batch_size, embedding_size), name='Y')
                # self._phase_train = tf.placeholder(tf.bool, 'Phase')

            with tf.variable_scope('Layers') as scope:
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=256)
                initial_state = state = lstm_cell.zero_state(batch_size=tf.shape(self._x)[0], dtype=tf.float32)

                for i in range(num_steps):
                    if i > 0: scope.reuse_variables()
                    output, state = lstm_cell(self._hot[:, i, :], state)

                final_output = output

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
