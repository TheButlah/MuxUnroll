from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import tensorflow as tf

from time import time, strftime


class LSTM(object):
    """This class implements the LSTM (Long Short-term Memory) architecture for Recurrent Neural Networks in TensorFlow.

    An LSTM is a type of Recurrent Neural Network that aims to minimize the effects of the Vanishing/Exploding Gradient
    problem by managing its internal cell state with forget, input, and output gates.
    """

    def __init__(self, embedding_size, num_steps, cell_size=128, time_major=True, bptt_method='traditional',
                 seed=None, load_model=None, config=None):
        """Initializes the architecture of the LSTM and returns an instance.

        Args:
            embedding_size: An integer that is equal to the size of the vectors used to embed the input elements.
                            Example: 10,000 for 10,000 unique words in the vocabulary
            num_steps:      An integer that is the number of unrolled steps that the LSTM takes. This is not (usually)
                            the length of the actual sequence. This number is related to the ability of the LSTM to
                            understand long-term dependencies in the data.
            cell_size:       An integer that is equal to the size of the LSTM cell. This is directly related to the
                            state size and number of parameters of the cell.
            time_major:     A boolean used to determine whether the first dimension of the data is the batch or time
                            dimension. Using time as the first dimension is more efficient.
            bptt_method:    A string that states the unrolling method for Back Propogation Through Time (BPTT) to use.
                            'traditional' uses python lists and tensorflow slicing to get the input for a given time.
                            'static' uses tf.nn.static_rnn. 'dynamic' uses tf.nn.dynamic_rnn.
            seed:           An integer used to seed the initial random state. Can be None to generate a new random seed.
            load_model:     If not None, then this should be a string indicating the checkpoint file containing data
                            that will be used to initialize the parameters of the model. Typically used when loading a
                            pre-trained model, or resuming a previous training session.
            config:         A ConfigProto that can configure the TensorFlow session. Only use if you know what it is.
        """
        print("Constructing Architecture...")
        self._embedding_size = embedding_size
        self._seed = seed
        self._num_steps = num_steps  # Tuples are used to ensure the dimensions are immutable
        self._cell_size = cell_size

        bptt_method = bptt_method.lower()
        if bptt_method not in ('traditional', 'dynamic', 'static'):
            raise ValueError("`bptt_method` must be one of 'traditional', 'dynamic', or 'static' ")

        self._last_time = 0  # Used by train to keep track of time
        self._iter_count = 0  # Used by train to keep track of iterations
        self._needs_update = False  # Used by train to indicate when enough time has passed to update summaries/stdout
        self._summary_writer = None  # Used by train to write summaries

        self._graph = tf.Graph()
        with self._graph.as_default():
            tf.set_random_seed(seed)

            with tf.variable_scope('Inputs'):
                x_shape = (num_steps, None) if time_major else (None, num_steps)
                y_shape = (None,)  # Batch

                # Initial variable values, only need to be passed when data changes (different batch)
                self._x_initial = tf.placeholder(tf.int32, shape=x_shape, name='X-Initial')
                self._y_initial = tf.placeholder(tf.int32, shape=y_shape, name='Y-Initial')

                # The collections=[] ensures that they do not get initialized with the other vars. Run self._init_inputs
                # any time the inputs change (typically on each new batch passed to self.train() or self.apply()
                self._x = tf.Variable(self._x_initial, trainable=False, collections=[], validate_shape=False, name='X')
                self._y = tf.Variable(self._y_initial, trainable=False, collections=[], validate_shape=False, name='Y')

                batch_size = tf.shape(self._x)[-1 if time_major else 0]  # Note that this is a scalar tensor
                self.batch_size = batch_size

                # Need to manually assign shape. Normally the variable constructor would do this already, but we needed
                # to disable it so that so we could dynamically change the shape when the model is trained/applied
                self._x.set_shape(x_shape)
                self._y.set_shape(y_shape)

                self._hot = tf.one_hot(indices=self._x, depth=embedding_size, name='Hot')  # X as one-hot encoded

            with tf.variable_scope('Unrolled') as scope:
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=cell_size)  # This defines the cell structure
                initial_state = lstm_cell.zero_state(batch_size=batch_size, dtype=tf.float32)  # Initial state

                self._sequence_lengths = tf.random_uniform(
                    shape=(batch_size,), minval=1, maxval=num_steps+1, dtype=tf.int32
                )  # , trainable=False, validate_shape=False, collections=[], name='Sequence-Lengths')

                def traditional_bptt():
                    """Calls the lstm cell with the state and output for each time until num_steps."""
                    state = initial_state
                    # Unroll the graph num_steps back into the "past"
                    for i in range(num_steps):
                        if i > 0: scope.reuse_variables()  # Reuse the variables created in the 1st LSTM cell
                        output, state = lstm_cell(  # Step the LSTM through the sequence
                            self._hot[i, ...] if time_major else self._hot[:, i, ...],
                            state
                        )
                    return output

                def dynamic_bptt():
                    """Uses dynamic_rnn to unroll the graph."""
                    outputs, states = tf.nn.dynamic_rnn(
                        lstm_cell, self._hot,
                        sequence_length=self._sequence_lengths,
                        initial_state=initial_state,
                        time_major=time_major,
                        scope=scope
                    )
                    return outputs[-1, ...] if time_major else outputs[:, -1, ...]

                def static_bptt():
                    """Uses static_rnn to unroll the graph"""
                    inputs = tf.unstack(self._hot, axis=0 if time_major else 1)

                    outputs, states = tf.nn.static_rnn(
                        lstm_cell, inputs,
                        sequence_length=self._sequence_lengths,
                        initial_state=initial_state,
                        scope=scope
                    )
                    return outputs[-1]

                # Emulate a switch statement
                final_output = {
                    'traditional': traditional_bptt,
                    'dynamic': dynamic_bptt,
                    'static': static_bptt
                }.get(bptt_method)()

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

                # Initialize or update per-batch Variables. Should be called whenever passing a new batch. This will
                # often be the case on each new call to train() or apply()
                with tf.control_dependencies([tf.variables_initializer([self._x, self._y], name='Init-Inputs')]):
                    self._init_batch = tf.no_op()  # tf.variables_initializer([self._sequence_lengths])

            with tf.variable_scope('Summaries'):
                tf.summary.scalar('Loss', self._loss)
                tf.summary.histogram('Weights', w)
                tf.summary.image('Weights', tf.expand_dims(tf.expand_dims(w, 0), -1))  # Visualize softmax weights
                self._summaries = tf.summary.merge_all()

            self._sess = tf.Session(config=config)
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
        up to the user, so that s/he can load the proper amount of data. Because the loss is averaged over  batch, a
        larger batch size will result in a more stable training process with potentially better results when applying
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

            # These will be passed into sess.run(), the first should be loss and last should be summaries.
            graph_elements = (self._loss, self._train_step, self._sequence_lengths, self._summaries)

            if log_dir is not None:
                if self._summary_writer is None:
                    print("Enabling Summaries!")
                    print("Run \"tensorboard --logdir=path/to/log-directory\" to view the summaries.")
                    self._summary_writer = tf.summary.FileWriter(log_dir, graph=self._graph)

            # Pass the initial values for X and Y in
            self._run_session(
                self._init_batch,
                feed_dict={self._x_initial: x_train, self._y_initial: y_train}
            )

            for epoch in range(num_epochs):
                # We need to decide whether to enable the summaries or not to save on computation
                if self._needs_update and log_dir is not None:
                    chosen_elements = None  # Include summaries
                else:
                    chosen_elements = -1  # Don't include summaries

                # Feed the data into the graph and run one step of the computation. Note that the training data
                # doesn't need to be re-fed once initially passed! This prevents having to waste time transferring data
                outputs = self._run_session(graph_elements[:chosen_elements])

                loss_val = outputs[0]  # Unpack the loss_val from the outputs

                if progress_info and self._needs_update:  # Only print progress when needed
                    print("Current Loss Value: %.10f, Percent Complete: %.4f" % (loss_val, epoch / num_epochs * 100))

                if self._needs_update and log_dir is not None:  # Only compute summaries when needed
                    self._summary_writer.add_summary(outputs[-1], self._iter_count)

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
            return self._run_session(self._y_hat, feed_dict={self._x: x_data})

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

    def _run_session(self, fetches, feed_dict=None, options=None, run_metadata=None):
        """Helper function to run the session. Useful for when enabling profiling."""
        return self._sess.run(fetches, feed_dict=feed_dict, options=options, run_metadata=run_metadata)
