from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf

from time import time, strftime
import os


class LSTM(object):
    """This class implements the LSTM (Long Short-term Memory) architecture for Recurrent Neural Networks in TensorFlow.

    An LSTM is a type of Recurrent Neural Network that aims to minimize the effects of the Vanishing/Exploding Gradient
    problem by managing its internal cell state with forget, input, and output gates.
    """

    def __init__(self, embedding_size, num_steps, seed=None, load_model=None):
        """Initializes the architecture of the LSTM and returns an instance.

        Args:
            embedding_size: An integer that is equal to the size of the vectors used to embed the input elements.
                            Example: 10,000 for 10,000 unique words in the vocabulary
            num_steps:      An integer that is the number of unrolled steps that the LSTM takes. This is not (usually)
                            the length of the actual sequence. This number is related to the ability of the LSTM to
                            understand long-term dependencies in the data.
            seed:           An integer used to seed the initial random state. Can be None to generate a new random seed.
            load_model:     If not None, then this should be a string indicating the checkpoint file containing data
                            that will be used to initialize the parameters of the model. Typically used when loading a
                            pre-trained model, or resuming a previous training session.
        """
        print("Constructing Architecture...")
        self._embedding_size = embedding_size
        self._seed = seed
        self._num_steps = num_steps  # Tuples are used to ensure the dimensions are immutable

        self._last_time = 0  # Used by train to keep track of time
        self._iter_count = 0  # Used by train to keep track of iterations
        self._needs_update = True  # Used by train to indicate when enough time has passed to update summaries/stdout
        self._summary_writer = None  # Used by train to write summaries

        self._graph = tf.Graph()
        with self._graph.as_default():
            tf.set_random_seed(seed)
            batch_size = None  # Although this variable is never modified, it is present to enhance code readability

            with tf.variable_scope('Inputs'):
                self._x = tf.placeholder(tf.int32, shape=(batch_size, num_steps), name='X')
                self._y = tf.placeholder(tf.int32, shape=(batch_size,), name='Y')

                self._hot = tf.one_hot(indices=self._x, depth=embedding_size, name='Hot')  # X as one-hot encoded

            with tf.variable_scope('Unrolled') as scope:
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=10)  # This defines the cell structure
                state = lstm_cell.zero_state(batch_size=tf.shape(self._x)[0], dtype=tf.float32)  # Initial state

                # Unroll the graph num_steps back into the "past"
                for i in range(num_steps):
                    if i > 0: scope.reuse_variables()  # Reuse the variables created in the 1st LSTM cell
                    output, state = lstm_cell(self._hot[:, i, :], state)  # Step the LSTM through the sequence
                final_output = output

            with tf.variable_scope('Softmax'):
                # Parameters
                w = tf.Variable(tf.random_normal((lstm_cell.output_size, embedding_size), stddev=0.1, name='Weights'))
                b = tf.Variable(tf.random_normal((embedding_size,), stddev=0.1, name='Bias'))
                scores = tf.matmul(final_output, w) + b  # The raw class scores to be fed into the loss function
                self._y_hat = tf.nn.softmax(scores, name='Y-Hat')  # Class probabilities, (batch_size, embedding_size)
                self._prediction = tf.argmax(self._y_hat, axis=1, name='Prediction')  # Vector of predicted classes

            with tf.variable_scope('Pipelining'):
                self._loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(  # Cross-entropy loss
                    logits=scores,
                    labels=self._y
                ), name='Loss')
                self._train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self._loss)  # Optimizer

            with tf.variable_scope('Summaries'):
                loss_summary = tf.summary.scalar('Loss', self._loss)
                self._summaries = tf.summary.merge_all()

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

    def train(self, x_train, y_train, num_epochs, start_stop_info=True, progress_info=True, log_dir=None):
        """Trains the model using the data provided as a batch.

        It is often infeasible to load the entire dataset into memory. For this reason, the selection of batches is left
        up to the user, so that s/he can load the proper amount of data. Because the loss is averaged over the batch,
        a larger batch size will result in a more stable training process with potentially better results when applying
        the model, although having a smaller batch size means less memory consumption and sometimes faster training.

        Args:
            x_train:  A numpy ndarray that contains the data to train over. Should should have a shape of
                [batch_size, num_steps]. Each element of this matrix should be the index of the item being trained on in
                its one-hot encoded representation. Indices are used instead of the full one-hot vector for efficiency.
            y_train:  A numpy ndarray that contains the labels that correspond to the data being trained on. Should have
                a shape of [batch_size]. Each element is the index of the on-hot encoded representation of the label.
            num_epochs:  The number of iterations over the provided batch to perform until training is considered to be
                complete. If all your data fits in memory and you don't need to mini-batch, then this should be a large
                number (>1000). Otherwise keep this small (<50) so the model doesn't become skewed by the small size of
                the provided mini-batch too quickly. It is expected that the code that selects the batch size will
                call this train method once with each new batch (or just once if mini-batching is not necessary)
            start_stop_info:  If true, print when the training begins and ends.
            progress_info:  If true, print what the current loss and percent completion over the course of training.
            log_dir:  If not None, then this should be a string indicating where to output log files for TensorBoard.

        Returns:
            The loss value after training
        """
        with self._sess.as_default():
            # Training loop for parameter tuning
            if start_stop_info: print("Starting training for %d epochs" % num_epochs)

            graph_elements = (self._loss, self._train_step, self._summaries)  # These will be passed into sess.run()
            num_elements = 2  # This will be used as the stop index when indexing graph_elements

            if log_dir is not None:
                if self._summary_writer is None:
                    print("Enabling Summaries!")
                    print("Run \"tensorboard --logdir=path/to/log-directory\" to view the summaries.")
                    self._summary_writer = tf.summary.FileWriter(log_dir, graph=self._graph)

            for epoch in range(num_epochs):
                # We need to decide whether to enable the summaries or not to save on computation
                if self._needs_update:
                    if log_dir is not None:
                        num_elements = 3  # Include summaries
                    else:
                        num_elements = 2  # Don't include summaries

                # Feed the data into the graph and run one step of the computation
                outputs = self._sess.run(graph_elements[:num_elements], feed_dict={self._x: x_train, self._y: y_train})
                loss_val = outputs[0]  # Unpack the loss_val from the outputs

                if progress_info and self._needs_update:  # Only print progress when needed
                    print("Current Loss Value: %.10f, Percent Complete: %.4f" % (loss_val, epoch / num_epochs * 100))

                if self._needs_update and log_dir is not None:  # Only compute summaries when needed
                    self._summary_writer.add_summary(outputs[2], self._iter_count)

                current_time = time()
                if (current_time - self._last_time) >= 5:  # Update logs/progress every 5 seconds
                    self._last_time = current_time
                    self._needs_update = True
                else:
                    self._needs_update = False

                self._iter_count += 1

            if start_stop_info: print("Completed Training.")
        return loss_val

    def apply(self, x_data):
        """Applies the model to the batch of data provided. Typically called after the model is trained.
        Args:
            x_data:  A numpy ndarray of the data to apply the model to. Should have the same shape as the training data.

        Returns:
            A numpy ndarray of the data, with shape [batch_size, embedding_size]. Rows are class probabilities.
            Example: result.shape is [batch_size, 100] when there are 100 unique words in the chosen dictionary.
        """
        with self._sess.as_default():
            return self._sess.run(self._y_hat, feed_dict={self._x: x_data})

    def save_model(self, save_path=None):
        """Saves the model in the specified file.
        Args:
            save_path:  The relative path to the file. By default, it is
                saved/LSTM-Year-Month-Date_Hour-Minute-Second.ckpt
        """
        with self._sess.as_default():
            print("Saving Model")
            if save_path is None:
                save_path = "saved/LSTM-%s.ckpt" % strftime("%Y-%m-%d_%H-%M-%S")
            dirname = os.path.dirname(save_path)
            if dirname is not '':
                os.makedirs(dirname, exist_ok=True)
            save_path = os.path.abspath(save_path)
            path = self._saver.save(self._sess, save_path)
            print("Model successfully saved in file: %s" % path)
