from model import LSTM

import numpy as np
from sklearn.model_selection import train_test_split


def main():
    seed = 1337
    batch_size = 1000
    sequence_length = 10
    embedding_size = 20
    np.random.seed(seed)
    model = LSTM(embedding_size, sequence_length-1, seed=seed)

    seq = np.arange(embedding_size)

    data = np.empty((batch_size, sequence_length))
    for i in range(batch_size):
        data[i] = np.roll(seq, i)[:sequence_length]

    print(data.shape)

    x = data[:, :-1]
    y = data[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.1, random_state=seed)
    model.train(x_train, y_train, num_epochs=5000)

    results = model.apply(x_train)
    print("Training Accuracy: ", np.equal(np.argmax(results, axis=1), y_train).astype(np.float32).mean())

    results = model.apply(x_test)
    print("Testing Accuracy: ", np.equal(np.argmax(results, axis=1), y_test).astype(np.float32).mean())

    print(model.apply(np.array([[1,2,3,4,5,6,7,8,9], [11,12,13,14,15,16,17,18,19], [15,16,17,18,19,0,1,2,3]])))


if __name__ == "__main__":
    main()