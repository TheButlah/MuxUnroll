from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np

from time import time, strftime
from collections import OrderedDict
from util import find_ge


class LSTM(object):
    """This class implements the LSTM (Long Short-term Memory) architecture for Recurrent Neural Networks in TensorFlow.

    An LSTM is a type of Recurrent Neural Network that aims to minimize the effects of the Vanishing/Exploding Gradient
    problem by managing its internal cell state with forget, input, and output gates.
    """

    def __init__(self, embedding_size, num_steps, cell_size=128, time_major=True,
                 seed=None, load_model=None, selected_steps=None,config=None):
        """Initializes the architecture of the LSTM and returns an instance.

        Args:
            embedding_size: An integer that is equal to the size of the vectors used to embed the input elements.
                            Example: 10,000 for 10,000 unique words in the vocabulary
            num_steps:      An integer that is the number of unrolled steps that the LSTM takes. No provided sequence
                            should be longer than this number. This number is related to the ability of the LSTM to
                            understand long-term dependencies in the data.
            cell_size:      An integer that is equal to the size of the LSTM cell. This is directly related to the
                            state size and number of parameters of the cell.
            time_major:     A boolean used to determine whether the first dimension of the data is the batch or time
                            dimension. Using time as the first dimension is more efficient.
            seed:           An integer used to seed the initial random state. Can be None to generate a new random seed.
            load_model:     If not None, then this should be a string indicating the checkpoint file containing data
                            that will be used to initialize the parameters of the model. Typically used when loading a
                            pre-trained model, or resuming a previous training session.
            selected_steps: A list (or iterable) of ints that are the timesteps at which to construct additional graphs
                            to multiplex between for the final output. For example, providing [1,25] with num_steps=100
                            results in the model automatically multiplexing between fully unrolled graphs with steps of
                            either 1, 25, or 100. This enables highly customizable control over the multiplexer. The
                            benefit of the multiplexer is that a faster unrolling method can be used for all the graphs,
                            yet whenever the maximum sequence length for a particular batch is small enough where using
                            the fully unrolled graph is slow, computation won't be wasted because a smaller graph is
                            selected by the multiplexer. Will throw a ValueError if any elements of this list are
                            greater than or equal to num_steps.
            config:         A ConfigProto that can configure the TensorFlow session. Only use if you know what it is.
        """
        print("Constructing Architecture...")
        self._embedding_size = embedding_size
        self._seed = seed
        self._num_steps = num_steps  # Tuples are used to ensure the dimensions are immutable
        self._cell_size = cell_size
        self._num_steps = num_steps
        self._time_major = time_major

        self._last_time = 0  # Used by train to keep track of time
        self._iter_count = 0  # Used by train to keep track of iterations
        self._needs_update = False  # Used by train to indicate when enough time has passed to update summaries/stdout
        self._summary_writer = None  # Used by train to write summaries

        self._multiplexer = OrderedDict()
        """
        dict of dicts. indexed by timestep (timestep is 1-index, not 0-indexed), then by one of:
        'y_hat', 'predictions', 'loss', 'train_step', 'same_length_init', 'var_length_init', 'summaries'
        """

        if selected_steps is not None:
            selected_steps = sorted(selected_steps)
            if not (0 <= selected_steps[0] < num_steps and 0 <= selected_steps[-1] < num_steps):
                raise ValueError("selected_steps must contain only integers in the range [1,num_steps)")
            for x in selected_steps:
                if not isinstance(x, int):
                    raise ValueError("selected_steps must contain only integers in the range [1,num_steps)")
        selected_steps.append(num_steps)

        self._graph = tf.Graph()
        with self._graph.as_default():
            tf.set_random_seed(seed)

            with tf.variable_scope('Inputs'):
                x_shape = (num_steps, None) if time_major else (None, num_steps)
                y_shape = (None,)  # Batch
                lengths_shape = (None,)

                # Initial variable values, only need to be passed when data changes (different batch)
                self._x_initial = tf.placeholder(tf.int32, shape=x_shape, name='X-Initial')
                self._y_initial = tf.placeholder(tf.int32, shape=y_shape, name='Y-Initial')
                self._lengths_initial = tf.placeholder(tf.int32, shape=lengths_shape, name='Lengths-Initial')

                # The collections=[] ensures that they do not get initialized with the other vars. Run self._init_inputs
                # any time the inputs change (typically on each new batch passed to self.train() or self.apply()
                self._x = tf.get_variable(
                    name='X', trainable=False, collections=[], validate_shape=False,
                    initializer=self._x_initial)
                self._y = tf.get_variable(
                    name='Y', trainable=False, collections=[], validate_shape=False,
                    initializer=self._y_initial)
                self._lengths = tf.get_variable(
                    name='Lengths', trainable=False, collections=[], validate_shape=False,
                    initializer=self._lengths_initial)

                # Need to manually assign shape. Normally the variable constructor would do this already, but we needed
                # to disable it so that so we could dynamically change the shape when the model is trained/applied
                self._x.set_shape(x_shape)
                self._y.set_shape(y_shape)
                self._lengths.set_shape(lengths_shape)

                self._batch_size = tf.shape(self._x)[-1 if time_major else 0]  # Note that this is a scalar tensor

                self._hot = tf.one_hot(indices=self._x, depth=embedding_size, name='Hot')  # X as one-hot encoded

            with tf.variable_scope('Unrolled') as scope:
                lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=cell_size)  # This defines the cell structure
                initial_state = state = lstm_cell.zero_state(batch_size=self._batch_size, dtype=tf.float32)

                # Unroll the graph num_steps back into the "past"
                self._outputs = []
                for i in range(num_steps):
                    if i > 0: scope.reuse_variables()  # Reuse the variables created in the 1st LSTM cell
                    output, state = lstm_cell(  # Step the LSTM through the sequence
                        (self._hot[i, ...] if time_major else self._hot[:, i, ...]), state)
                    self._outputs.append(output)  # python list

            def construct_output_graph(time_step):
                """Constructs the rest of the graph that uses a given timestep as its input.

                Args:
                    time_step: An int that will be used to select the tensor to use as input to the rest of the graph.
                    Note that 1 means 1st timestep (length = 1) because we aren't 0-indexing
                """
                final_output = self._outputs[time_step-1]  # We may not be 0-indexing, but python sure is!
                results = {}
                # use name_scope instead of variable_scope so that it only applies the custom name to ops
                with tf.name_scope(str(time_step)):
                    with tf.variable_scope('Softmax'):
                        # Parameters
                        w = tf.get_variable(
                            name='Weights',
                            initializer=tf.random_normal((lstm_cell.output_size, embedding_size), stddev=0.1))
                        b = tf.get_variable(
                            name='Bias',
                            initializer=tf.random_normal((embedding_size,), stddev=0.1))
                        scores = tf.matmul(final_output, w) + b  # The raw class scores to be fed into the loss function
                        results['y_hat'] = y_hat = tf.nn.softmax(
                            scores, name='Y-Hat')  # Class probabilities, (batch_size, embedding_size)
                        results['predictions'] = tf.argmax(y_hat, axis=1, name='Predictions')  # Vector of classes

                    with tf.variable_scope('Pipelining'):
                        results['loss'] = loss = tf.reduce_mean(
                            tf.nn.sparse_softmax_cross_entropy_with_logits(  # Cross-entropy loss
                                logits=scores,
                                labels=self._y),
                            name='Loss')
                        results['train_step'] = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)  # Optimizer

                        # Initialize or update per-batch input vars.
                        same_length_init = tf.variables_initializer([self._x, self._y], name='SameLength-Init')
                        with tf.control_dependencies([same_length_init]):
                            var_length_init = tf.variables_initializer([self._lengths], name='VarLength-Init')

                        # Initialize batch for constant length = num_steps. Call whenever passing a new batch.
                        # This will often be the case on each new call to train() or apply()
                        with tf.control_dependencies([same_length_init]):
                            # Have to use initialized_value() because tf cant understand control flow for uninit values
                            is_input_steps_correct = tf.equal(
                                tf.shape(self._x.initialized_value())[0 if time_major else -1],
                                num_steps)
                            results['same_length_init'] = is_input_steps_correct

                        # Initialize batch for variable length. Should be called whenever passing a new batch. This will
                        # often be the case on each new call to train() or apply()
                        with tf.control_dependencies([var_length_init]):
                            is_length_steps_correct = tf.less_equal(tf.reduce_max(self._lengths.initialized_value()),
                                                                    num_steps)
                            results['var_length_init'] = tf.logical_and(is_input_steps_correct, is_length_steps_correct)

                    with tf.variable_scope('Summaries'):
                        tf.summary.scalar('Loss', results['loss'])
                        tf.summary.histogram('Weights', w)
                        tf.summary.image('Weights', tf.expand_dims(tf.expand_dims(w, 0), -1))  # softmax weights
                        results['summaries'] = tf.summary.merge_all()
                return results

            with tf.variable_scope('Multiplexer') as scope:
                initialized = False
                for time_step in selected_steps:
                    if initialized:
                        scope.reuse_variables()
                    else:
                        initialized = True
                    self._multiplexer[time_step] = construct_output_graph(time_step)

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

    def train(self, x_train, y_train, lengths_train=None, num_epochs=1000, start_stop_info=True, progress_info=True,
              log_dir=None):
        """Trains the model using the data provided as a batch.

        It is often infeasible to load the entire dataset into memory. For this reason, the selection of batches is left
        up to the user, so that s/he can load the proper amount of data. Because the loss is averaged over  batch, a
        larger batch size will result in a more stable training process with potentially better results when applying
        the model, although having a smaller batch size means less memory consumption and sometimes faster training.

        Args:
            x_train:  A numpy ndarray that contains the data to train over. Should should have a shape of
                [num_steps, batch_size] if _time_major=True, otherwise [batch_size, num_steps]. Each element of this
                matrix should be the index of the item being trained on in its one-hot encoded representation. Indices
                are used instead of the full one-hot vector for efficiency.
            y_train:  A numpy ndarray that contains the labels that correspond to the data being trained on. Should have
                a shape of [batch_size]. Each element is the index of the on-hot encoded representation of the label.
            lengths_train:  A numpy ndarray that contains the sequence length for each element in the batch. If none,
                training speed is a lot faster, and sequence lengths are assumed to be the full length of the time dim.
            num_epochs:  The number of iterations over the provided batch to perform until training is considered to be
                complete. If all your data fits in memory and you don't need to mini-batch, then this should be a large
                number (>=1000). Otherwise keep this small (<50) so the model doesn't become skewed by the small size of
                the provided mini-batch too quickly. It is expected that the code that selects the batch size will
                call this train method once with each new batch (or just once if mini-batching is not necessary)
            start_stop_info:  If true, print when the training begins and ends.
            progress_info:  If true, print what the current loss and percent completion over the course of training.
            log_dir:  If not None, then this should be a string indicating where to output log files for TensorBoard.

        Returns:
            The loss value after training
        """
        with self._sess.as_default():

            if log_dir is not None:
                if self._summary_writer is None:
                    print("Enabling Summaries!")
                    print("Run \"tensorboard --logdir=path/to/log-directory\" to view the summaries.")
                    self._summary_writer = tf.summary.FileWriter(log_dir, graph=self._graph)

            # Pass in the initial values for X and Y, and if specified, sequence_lengths
            if lengths_train is None:
                index = self._num_steps
                selected_graph = self._multiplexer[index]
                init_success = self._run_session(
                    selected_graph['same_length_init'],
                    feed_dict={self._x_initial: x_train, self._y_initial: y_train})
            else:
                max_sequence_length = np.max(lengths_train)
                index = find_ge(self._multiplexer.keys(), max_sequence_length)
                selected_graph = self._multiplexer[index]
                init_success = self._run_session(
                    selected_graph['var_length_init'],
                    feed_dict={self._x_initial: x_train, self._y_initial: y_train,
                               self._lengths_initial: lengths_train})
                if start_stop_info: print("Max Sequence Length: ", max_sequence_length)

            if not init_success:
                raise ValueError("Initialization failed! Make sure your sequence lengths and time dimension are "
                                 "correctly set with respect to `num_steps`.")

            # These will be passed into sess.run(), the first should be 'loss' and last should be 'summaries'.
            fetches = ('loss', 'train_step', 'summaries')
            graph_elements = [selected_graph[key] for key in fetches]

            if start_stop_info: print("Starting training for %d epochs using %d unroll steps" % (num_epochs, index))

            # Training loop for the given batch
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

    def apply(self, x_data, lengths_data=None, time_major=True):
        """Applies the model to the batch of data provided. Typically called after the model is trained.

        Args:
            x_data:  An ndarray of the data to apply the model to. Should have a similar shape to the training data.
                The time dimension may be smaller than `num_steps` if it makes sense within the context of the problem.
                If `time_major` varies from what was supplied in the constructor, then the data will be transposed
                before being fed to the model. For efficiency, it is therefore best to have data in the same format as
                the model.
            lengths_data:  An optional numpy ndarray that describes the sequence length of each element in the batch.
                Should be a vector the length of the batch size. If None, then sequence length is assumed to be the same
                for each batch element and will be the length of the time dimension. If it is None, then the size of the
                time dimension of `x_data` will be used.

        Returns:
            A numpy ndarray of the data, with shape [batch_size, embedding_size]. Rows are class probabilities.
            Example: result.shape is [batch_size, 100] when there are 100 unique words in the chosen dictionary.
        """
        with self._sess.as_default():
            if not (time_major is self._time_major):
                x_data = x_data.T
            if lengths_data is None:
                index = x_data.shape[0 if time_major else 1]
                selected_graph = self._multiplexer[index]
                y_hat = selected_graph['y_hat']
                return self._run_session(y_hat, feed_dict={self._x: x_data})
            else:
                max_sequence_length = np.max(lengths_data)
                index = find_ge(self._multiplexer.keys(), max_sequence_length)
                selected_graph = self._multiplexer[index]
                y_hat = selected_graph['y_hat']
                return self._run_session(y_hat, feed_dict={self._x: x_data, self._lengths: lengths_data})

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
