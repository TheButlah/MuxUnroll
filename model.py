from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import tensorflow as tf

from time import time, strftime
from tensorflow.contrib.rnn import LSTMStateTuple


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
            num_steps:      An integer that is the number of unrolled steps that the LSTM takes. No provided sequence
                            should be longer than this number. This number is related to the ability of the LSTM to
                            understand long-term dependencies in the data.
            cell_size:      An integer that is equal to the size of the LSTM cell. This is directly related to the
                            state size and number of parameters of the cell.
            time_major:     A boolean used to determine whether the first dimension of the data is the batch or time
                            dimension. Using time as the first dimension is more efficient.
            bptt_method:    A string that states the unrolling method for Back Propogation Through Time (BPTT) to use.
                            'traditional' uses python lists and tensorflow slicing to get the input for a given time.
                            'dynamic' uses tf.nn.dynamic_rnn.
                            'static' uses tf.nn.static_rnn.
                            'loop' uses a tf.while_loop and python lists
                            'array' uses a python loop and TensorArrays
                            'stack' uses a python loop and tf.unstack
                            'where' is same as 'traditional' but uses tf.where to mask the output
                            'masked' is the same as 'traditional' but uses a Tensor as a mask multiplied to the output.
                            'custom' is the best attempt at a fully custom dynamically unrolled yet efficient bptt
                            implementation. It has the ability to switch between a fixed-length and variable-length
                            graph at the python level.
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
        if bptt_method not in ('traditional', 'dynamic', 'static', 'loop', 'array', 'stack', 'where', 'masked',
                               'custom'):
            raise ValueError("`bptt_method` must be one of 'traditional', 'dynamic', 'static', "
                             "'loop', 'array', 'stack', 'where', 'masked', or 'custom'.")

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
                lengths_shape = (None,)

                # Initial variable values, only need to be passed when data changes (different batch)
                self._x_initial = tf.placeholder(tf.int32, shape=x_shape, name='X-Initial')
                self._y_initial = tf.placeholder(tf.int32, shape=y_shape, name='Y-Initial')
                self._lengths_initial = tf.placeholder(tf.int32, shape=lengths_shape, name='Lengths-Initial')

                # The collections=[] ensures that they do not get initialized with the other vars. Run self._init_inputs
                # any time the inputs change (typically on each new batch passed to self.train() or self.apply()
                self._x = tf.Variable(self._x_initial, trainable=False, collections=[], validate_shape=False, name='X')
                self._y = tf.Variable(self._y_initial, trainable=False, collections=[], validate_shape=False, name='Y')
                self._lengths = tf.Variable(self._lengths_initial, trainable=False, collections=[],
                                            validate_shape=False, name='Lengths')

                # Need to manually assign shape. Normally the variable constructor would do this already, but we needed
                # to disable it so that so we could dynamically change the shape when the model is trained/applied
                self._x.set_shape(x_shape)
                self._y.set_shape(y_shape)
                self._lengths.set_shape(lengths_shape)

                self._max_sequence_length = tf.reduce_max(self._lengths)
                self._batch_size = tf.shape(self._x)[-1 if time_major else 0]  # Note that this is a scalar tensor

                self._hot = tf.one_hot(indices=self._x, depth=embedding_size, name='Hot')  # X as one-hot encoded

            with tf.variable_scope('Unrolled') as scope:
                lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=cell_size)  # This defines the cell structure
                initial_state = lstm_cell.zero_state(batch_size=self._batch_size, dtype=tf.float32)  # Initial state
                self._outputs = None

                def traditional_bptt():
                    """Uses tensor slicing and python loop to call lstm_cell for each timestep. Does not mask."""
                    state = initial_state

                    # Unroll the graph num_steps back into the "past"
                    self._outputs = []
                    for i in range(num_steps):
                        if i > 0: scope.reuse_variables()  # Reuse the variables created in the 1st LSTM cell
                        output, state = lstm_cell(  # Step the LSTM through the sequence
                            (self._hot[i, ...] if time_major else self._hot[:, i, ...]), state)
                        self._outputs.append(output)  # python list

                    self._outputs = tf.stack(self._outputs)  # Tensor, shape=(num_steps, batch_size, embedding_size)
                    return output

                def masked_bptt():
                    """Same as traditional, but properly masks the outputs using multiplication."""
                    state = initial_state

                    # Unroll the graph num_steps back into the "past"
                    self._outputs = []
                    for i in range(num_steps):
                        if i > 0: scope.reuse_variables()  # Reuse the variables created in the 1st LSTM cell
                        output, state = lstm_cell(  # Step the LSTM through the sequence
                            (self._hot[i, ...] if time_major else self._hot[:, i, ...]), state)
                        self._outputs.append(output)  # python list

                    self._outputs = tf.stack(self._outputs)  # Tensor, shape=(num_steps, batch_size, embedding_size)
                    mask = tf.sequence_mask(self._lengths, maxlen=num_steps, dtype=tf.float32)  # (batch_size, num_steps)
                    mask = tf.expand_dims(mask, axis=-1)  # (batch_size, num_steps, 1)
                    mask = tf.transpose(mask, (1, 0, 2))  # (num_steps, batch_size, 1)
                    self._outputs *= mask
                    return self._outputs[-1]

                def where_bptt():
                    """Same as `traditional_bptt`, but uses tf.where to mask the output and state."""
                    state = initial_state
                    zeros = tf.zeros((self._batch_size, lstm_cell.output_size))

                    # Unroll the graph num_steps back into the "past"
                    self._outputs = []
                    for i in range(num_steps):
                        if i > 0: scope.reuse_variables()  # Reuse the variables created in the 1st LSTM cell
                        output, new_state = lstm_cell(  # Step the LSTM through the sequence
                            (self._hot[i, ...] if time_major else self._hot[:, i, ...]), state)

                        cond = tf.less(i, self._lengths)
                        output = tf.where(cond, x=output, y=zeros, name='Passed-Output')
                        state = [tf.where(cond, new_state, state, name='Passed-State')
                                 for new_state, state in zip(new_state, state)]  # Mask each element of LSTMStateTuple
                        self._outputs.append(output)  # python list

                    self._outputs = tf.stack(self._outputs)  # Tensor, shape=(num_steps, batch_size, embedding_size)
                    return output

                def dynamic_bptt():
                    """Uses dynamic_rnn to unroll the graph."""
                    outputs, states = tf.nn.dynamic_rnn(
                        lstm_cell, self._hot,
                        sequence_length=self._lengths,
                        initial_state=initial_state,
                        time_major=time_major,
                        scope=scope
                    )
                    self._outputs = outputs  # Tensor
                    return outputs[-1, ...] if time_major else outputs[:, -1, ...]

                def static_bptt():
                    """Uses static_rnn to unroll the graph"""
                    inputs = tf.unstack(self._hot, axis=0 if time_major else 1)

                    outputs, states = tf.nn.static_rnn(
                        lstm_cell, inputs,
                        sequence_length=self._lengths,
                        initial_state=initial_state,
                        scope=scope
                    )

                    self._outputs = tf.stack(outputs)  # Tensor, shape=(num_steps, batch_size, embedding_size)
                    return outputs[-1]

                def loop_bptt():
                    """Uses tf.while_loop to unroll graph, but doesn't use TensorArrays or tf.where.

                    Note: because of how tf.while_loop works, it can only accumulate the list of outputs over the
                    various timesteps using TensorFlow data structures, hence `loop_bppt` does not have a
                    `self._outputs` variable.
                    """
                    # Initial time, used as a counter variable in the loop for the unrolling
                    time = tf.constant(0, dtype=tf.int32, name='Time')

                    def loop_body(time, _, state_t):
                        new_output, new_state = lstm_cell(  # Step the LSTM through the sequence
                            self._hot[time, ...] if time_major else self._hot[:, time, ...],
                            state_t
                        )

                        return time+1, new_output, new_state

                    _, last_output, _ = tf.while_loop(
                        lambda time, *unused: time < self._max_sequence_length,
                        loop_body,
                        loop_vars=(time, tf.zeros((self._batch_size, lstm_cell.output_size)), initial_state),
                        shape_invariants=(
                            time.shape,
                            tf.TensorShape((None, lstm_cell.output_size)),
                            LSTMStateTuple(*tuple(tf.TensorShape((None, size)) for size in lstm_cell.state_size)),
                        )
                    )
                    return last_output

                def array_bptt():
                    self._outputs = []

                    input_ta = tf.TensorArray(tf.float32, size=num_steps, tensor_array_name='Inputs')
                    input_ta = input_ta.unstack(self._hot if time_major else tf.transpose(self._hot, (1,0,2)))

                    state = initial_state
                    # Unroll the graph num_steps back into the "past"
                    for i in range(num_steps):
                        if i > 0: scope.reuse_variables()  # Reuse the variables created in the 1st LSTM cell
                        output, state = lstm_cell(  # Step the LSTM through the sequence
                            input_ta.read(i), state)
                        self._outputs.append(output)  # python list
                    return output

                def stack_bptt():
                    """Almost the same as traditional, but uses tf.unstack"""
                    inputs = tf.unstack(self._hot, axis=0 if time_major else 1)
                    state = initial_state

                    self._outputs = []

                    for i, x in enumerate(inputs):
                        if i > 0: scope.reuse_variables()  # Reuse the variables created in the 1st LSTM cell
                        output, state = lstm_cell(x, state)  # Step the LSTM through the sequence
                        self._outputs.append(output)  # python list
                    return output

                def custom_bptt():

                    def dynamic_graph():
                        # Initial time, used as a counter variable in the loop for the unrolling
                        time = tf.constant(0, dtype=tf.int32, name='Time')
                        zeros = tf.zeros((self._batch_size, lstm_cell.output_size))

                        input_ta = tf.TensorArray(
                            tf.float32, size=self._max_sequence_length, tensor_array_name='Inputs')
                        output_ta = tf.TensorArray(
                            tf.float32, size=self._max_sequence_length, tensor_array_name='Outputs')
                        input_ta = input_ta.unstack((self._hot if time_major else
                                                     tf.transpose(self._hot, (1, 0, 2)))[:self._max_sequence_length])

                        def loop_body(time, output_ta, state):
                            input_t = input_ta.read(time, name='Input')
                            new_output, new_state = lstm_cell(input_t, state)

                            '''cond = time < self._lengths  # Boolean Tensor, shape=(batch_size,)
                            # Zero out when sequence is over
                            new_output = tf.where(cond, x=new_output, y=zeros, name='Zeroed-Output')
                            # Pass forward state for each element of LSTMStateTuple when sequence is over
                            new_state = LSTMStateTuple(*(tf.where(cond, new_state, state, name='Passed-State')
                                                       for new_state, state in zip(new_state, state)))'''

                            return time + 1, output_ta.write(time, new_output), new_state

                        _, final_output_ta, final_state = tf.while_loop(
                            lambda time, *unused: time < self._max_sequence_length,
                            loop_body,
                            loop_vars=(time, output_ta, initial_state)

                        )
                        outputs = final_output_ta.stack()
                        mask = tf.sequence_mask(self._lengths, maxlen=self._max_sequence_length,
                                                dtype=tf.float32)  # (batch_size, num_steps)
                        mask = tf.expand_dims(mask, axis=-1)  # (batch_size, num_steps, 1)
                        mask = tf.transpose(mask, (1, 0, 2))  # (num_steps, batch_size, 1)
                        outputs *= mask
                        return outputs

                    self._outputs = dynamic_graph()
                    return self._outputs[-1]

                # Emulate a switch statement
                final_output = {
                    'traditional': traditional_bptt,
                    'dynamic': dynamic_bptt,
                    'static': static_bptt,
                    'loop': loop_bptt,
                    'array': array_bptt,
                    'stack': stack_bptt,
                    'where': where_bptt,
                    'masked': masked_bptt,
                    'custom': custom_bptt,
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

                # Initialize or update per-batch input vars.
                same_length_init = tf.variables_initializer([self._x, self._y], name='SameLength-Init')
                with tf.control_dependencies([same_length_init]):
                    var_length_init = tf.variables_initializer([self._lengths], name='VarLength-Init')

                # Initialize batch for constant length = num_steps. Should be called whenever passing a new batch.
                # This will often be the case on each new call to train() or apply()
                with tf.control_dependencies([same_length_init]):
                    # Have to use initialized_value() because tf cant understand control flow for uninitialized values
                    is_input_steps_correct = tf.equal(
                        tf.shape(self._x.initialized_value())[0 if time_major else -1],
                        num_steps)
                    self._same_length_init = is_input_steps_correct

                # Initialize batch for variable length. Should be called whenever passing a new batch. This will
                # often be the case on each new call to train() or apply()
                with tf.control_dependencies([var_length_init]):
                    is_length_steps_correct = tf.less_equal(tf.reduce_max(self._lengths.initialized_value()), num_steps)
                    self._var_length_init = tf.logical_and(is_input_steps_correct, is_length_steps_correct)

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

    def train(self, x_train, y_train, lengths_train=None, num_epochs=1000, start_stop_info=True, progress_info=True,
              log_dir=None):
        """Trains the model using the data provided as a batch.

        It is often infeasible to load the entire dataset into memory. For this reason, the selection of batches is left
        up to the user, so that s/he can load the proper amount of data. Because the loss is averaged over  batch, a
        larger batch size will result in a more stable training process with potentially better results when applying
        the model, although having a smaller batch size means less memory consumption and sometimes faster training.

        Args:
            x_train:  A numpy ndarray that contains the data to train over. Should should have a shape of
                [num_steps, batch_size] if time_major=True, otherwise [batch_size, num_steps]. Each element of this
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
            # Training loop for a given batch
            if start_stop_info: print("Starting training for %d epochs" % num_epochs)

            # These will be passed into sess.run(), the first should be loss and last should be summaries.
            graph_elements = (self._loss, self._train_step, self._summaries)

            if log_dir is not None:
                if self._summary_writer is None:
                    print("Enabling Summaries!")
                    print("Run \"tensorboard --logdir=path/to/log-directory\" to view the summaries.")
                    self._summary_writer = tf.summary.FileWriter(log_dir, graph=self._graph)

            # Pass in the initial values for X and Y, and if specified, sequence_lengths
            if lengths_train is None:
                init_success = self._run_session(
                    self._same_length_init,
                    feed_dict={self._x_initial: x_train, self._y_initial: y_train})
            else:
                init_success = self._run_session(
                    self._var_length_init,
                    feed_dict={self._x_initial: x_train, self._y_initial: y_train,
                               self._lengths_initial: lengths_train})
                print("Max Sequence Length: ", self._run_session(self._max_sequence_length))

            if not init_success:
                raise ValueError("Initialization failed! Make sure your sequence lengths and time dimension are "
                                 "correctly set with respect to `num_steps`.")

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

    def apply(self, x_data, lengths_data=None):
        """Applies the model to the batch of data provided. Typically called after the model is trained.
        Args:
            x_data:  An ndarray of the data to apply the model to. Should have a similar shape to the training data.
                Depending on the value of `bptt_method`, the time dimension may or may not be permitted to be larger or
                smaller than `num_steps`.
            lengths_data:  An optional numpy ndarray that describes the sequence length of each element in the batch.
                Should be a vector the length of the batch size. If None, then sequence length is assumed to be the same
                for each batch element and will be the length of the time dimension.

        Returns:
            A numpy ndarray of the data, with shape [batch_size, embedding_size]. Rows are class probabilities.
            Example: result.shape is [batch_size, 100] when there are 100 unique words in the chosen dictionary.
        """
        with self._sess.as_default():
            if lengths_data is None:
                return self._run_session(self._y_hat, feed_dict={self._x: x_data})
            else:
                return self._run_session(self._y_hat, feed_dict={self._x: x_data, self._lengths: lengths_data})

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
