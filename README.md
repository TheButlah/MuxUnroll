# LSTM
An implementation of the LSTM Recurrent Neural Network in TensorFlow using Multiplexer Unrolling.

Unlike a regular RNN, this implementation of an LSTM uses a technique I developed called "Multiplexer Unrolling". It is a technique used when performing the Backpropogation Through Time (BPTT) algorithm. During BPTT, the graph is "unrolled" meaning it ceases to be a RNN and instead becomes a feed-forward neural network. In practice, the graph cannot be unrolled to an infinite length, so it must be truncated. In TensorFlow, the way this is done is to unroll to the maximum length of the sequence. This is what i coin as "traditional unrolling".

The problem with traditional unrolling is that there is a lot of wasted computation when your input sequence is less than the full unrolling length. TensorFlow has attempted to combat this with its `tf.nn.dynamic_rnn` method, but it performs very poorly, typically over twice as poorly as traditional unrolling. This is due to the slowness of the `tf.while_loop` operation in TensorFlow.

My method instead creates many duplicate graphs with shared weights, each with a different unrolling length. It then multiplexes between them to determine the best one to use for a given input batch. This takes more resources to construct the graph at initialization, but once constructed training speeds are dramatically faster. **In my tests, I managed to get an average speedup of ~2x faster training speeds** in graphs with maximum unroll lengths of 100 but sequences that averaged 50 units long. 
