from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np

from sklearn.model_selection import train_test_split
from time import time
from model import LSTM


def main():
    seed = 1337
    batch_size = 1000
    num_epochs = 10000
    backprop_steps = sequence_length = 10  # In our architecture, the graph is unfolded backprop_steps, which means that
    embedding_size = 10                    # we need to pass 10 unique datapoints into the graph (one for each step)
    np.random.seed(seed)
    model = LSTM(embedding_size, backprop_steps-1, seed=seed)

    data = np.empty((batch_size, backprop_steps))
    for i in range(batch_size):
        data[i] = np.random.choice(10, size=(10,), replace=False)

    x = data[:, :-1]
    y = data[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=seed)
    model.train(x_train, y_train, num_epochs=num_epochs)

    print_examples(model, np.array(
        [[1,2,3,4,5,6,7,8,9]
        ,[4,5,6,7,8,9,0,1,2]
        ,[9,8,7,6,5,4,3,2,1]]
    ))

    results = model.apply(x_train)
    print("Training Accuracy: ", np.equal(np.argmax(results, axis=1), y_train).astype(np.float32).mean())

    results = model.apply(x_test)
    print("Testing Accuracy: ", np.equal(np.argmax(results, axis=1), y_test).astype(np.float32).mean())


def print_examples(model, batch, print_start_stop=True, print_timing=True):
    if print_start_stop: print("Applying model...")
    start_time = time()
    results = model.apply(batch)
    elapsed = time() - start_time
    if print_start_stop: print("Finished!")
    for x, y in zip(batch, results):
        print("\nInput:")
        print(x)
        print("Output:")
        print(y)
    if print_timing: print("\nInference Time: %f seconds" % elapsed)


if __name__ == "__main__":
    main()