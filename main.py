from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import os
import tensorflow as tf

from sklearn.model_selection import train_test_split
from model import LSTM
from util import print_examples
from time import time
import math


def main():
    time_major = True
    seed = 1337
    batch_size = 5000
    num_epochs = 1000
    sequence_length = 100
    # In our architecture, the graph is unfolded backprop_steps
    backprop_steps = sequence_length-1
    embedding_size = 10  # The number of unique elements in our "Vocabulary", in this case all 10 digits.

    np.random.seed(seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True  # Make it so that the program does not grab all GPUs' memory at start

    # Initialize the model architecture, but do not pass data
    model = LSTM(embedding_size, backprop_steps, cell_size=embedding_size, time_major=time_major,
                 selected_steps=(1,2,9,98), seed=seed, config=config)

    # Generate random permutations of base `embedding_size` digits, with the output being the one that did not appear.
    # For this problem to make any sense, sequence_length should be equal to embedding_size
    data = np.empty((batch_size, sequence_length))
    for i in range(batch_size):
        perm = np.random.choice(embedding_size, size=(embedding_size,), replace=False)
        data[i, :embedding_size-1] = perm[:-1]
        data[i, -1] = perm[-1]

    # Separate data and labels
    x = data[:, :-1]  # Shape: (batch_size, backprop_steps)
    y = data[:, -1]  # Shape: (batch_size)

    lengths = np.full((batch_size,), embedding_size-1, dtype=np.int32)

    datasets = train_test_split(  # Split into training and testing sets
        x, y, lengths, train_size=0.2, random_state=seed
    )

    explicit_examples = np.array(  # Apply the model to several example inputs
        [[1,2,3,4,5,6,7,8,9]
        ,[4,5,6,7,8,9,0,1,2]
        ,[9,8,7,6,5,4,3,2,1]]
    )

    x_train, x_test, y_train, y_test, lengths_train, lengths_test, explicit_examples = format_for_model(
        datasets + [explicit_examples],
        should_transpose=time_major  # if _time_major, then transpose batch and time dims to be _time_major
    )

    start_time = time()
    # This actually trains the model on a single batch, which in our case is the entirety of the training data.
    model.train(x_train, y_train, lengths_train, num_epochs=num_epochs)
    elapsed_time = time() - start_time

    print("Training Duration: ", elapsed_time)

    # print_examples(model, explicit_examples)

    results = model.apply(x_train, lengths_train)
    print("Training Accuracy: ", np.equal(np.argmax(results, axis=1), y_train).astype(np.float32).mean())

    results = model.apply(x_test, lengths_test)
    print("Testing Accuracy: ", np.equal(np.argmax(results, axis=1), y_test).astype(np.float32).mean())


def format_for_model(the_list, should_transpose=False):
    """Helper function to transpose the batch and time dims of each element in `the_list`"""
    return [x.T for x in the_list] if should_transpose else the_list

if __name__ == "__main__":
    main()
