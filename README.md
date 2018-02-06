# MuxUnroll
An implementation of my Multiplexer Unrolling technique implemented for a LSTM Recurrent Neural Network in TensorFlow.

Unlike a regular RNN, this implementation of an LSTM uses a technique I developed called "Multiplexer Unrolling". It is a technique used when performing the Backpropogation Through Time (BPTT) algorithm. During BPTT, the graph is "unrolled" meaning it ceases to be a RNN and instead becomes a feed-forward neural network. In practice, the graph cannot be unrolled to an infinite length, so it must be truncated. In TensorFlow, the way this is done is to unroll to the maximum length of the sequence. This is what I call "traditional unrolling".

The problem with traditional unrolling is that there is a lot of wasted computation when your input sequence is less than the full unrolling length. TensorFlow has attempted to combat this with its `tf.nn.dynamic_rnn` and `tf.nn.static_rnn` methods, but they perform very poorly, typically over twice as poorly as traditional unrolling. This is due to the slowness of the `tf.while_loop` operation in TensorFlow and the fact that `tf.cond` still has to compute all dependencies of the conseqents of the statement, even if they will not be called.

My method instead creates many duplicate graphs with shared weights, each with a different unrolling length. It then multiplexes between them outside of TensorFlow to determine the best one to use for a given input batch. This takes more resources to construct the graph at initialization, but once constructed training speeds are dramatically faster. **In my tests, I managed to get an average speedup of ~2x faster training speeds** in graphs with maximum unroll lengths of 100 but sequences that averaged 50 units long.

Important Caveats:
- The maximum sequence length of a batch is what the multiplexer uses as the length. Hence try to minimize the longest length in each batch by sorting your data by length when batching. Note that this can add bias into the model so be smart about this.
- The largest benefits will be seen when using long maximum unrolling lengths.
- The graph construction can consume lots of memory. Be conservative with how many graphs you wish to construct.
