from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from model import LSTM
from util import print_examples
from tensorflow.python.client import timeline


def main():
    seed = 1337
    batch_size = 1000
    num_epochs = 10000
    sequence_length = 10
    # In our architecture, the graph is unfolded backprop_steps
    backprop_steps = sequence_length-1
    embedding_size = 10  # The number of unique elements in our "Vocabulary", in this case all 10 digits.

    np.random.seed(seed)

    config = None
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth=True  # Make it so that the program does not grab all GPUs' memory at start

    model = LSTM(embedding_size, backprop_steps, seed=seed, config=config)  # Initialize the model architecture, but do not pass data

    # Generate the input as a random arrangement of 9 digits, with the output being the one digit that did not appear
    data = np.empty((batch_size, sequence_length))
    for i in range(batch_size):
        data[i] = np.random.choice(10, size=(10,), replace=False)

    # Separate data and labels
    x = data[:, :-1]  # Shape: (batch_size, backprop_steps)
    y = data[:, -1]  # Shape: (batch_size)

    x_train, x_test, y_train, y_test = train_test_split(  # Split into training and testing sets
        x, y, train_size=0.2, random_state=seed
    )

    # This actually trains the model on a single batch, which in our case is the entirety of the training data.
    model.train(x_train, y_train, num_epochs=num_epochs, log_dir='logs/')

    print_examples(model, np.array(  # Apply the model to several example inputs
        [[1,2,3,4,5,6,7,8,9]
        ,[4,5,6,7,8,9,0,1,2]
        ,[9,8,7,6,5,4,3,2,1]]
    ))

    results = model.apply(x_train)
    print("Training Accuracy: ", np.equal(np.argmax(results, axis=1), y_train).astype(np.float32).mean())

    results = model.apply(x_test)
    print("Testing Accuracy: ", np.equal(np.argmax(results, axis=1), y_test).astype(np.float32).mean())


if __name__ == "__main__":
    main()
