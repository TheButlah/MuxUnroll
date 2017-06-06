from model import LSTM

import numpy as np
from sklearn.model_selection import train_test_split


def main():
    seed = 1337
    model = LSTM(10, 10, seed=seed)
    np.random.seed(seed)

    x = np.empty((500, 10))
    for i in range(x.shape[0]):
        x[i] = np.random.choice(10, size=(10,), replace=True)

    y = x.argmax(axis=1)

    print(x)
    print(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.75, random_state=seed)
    model.train(x_train, y_train, num_epochs=10000)

    results = model.apply(x_train)
    print("Training Accuracy: ", np.equal(np.argmax(results, axis=1), y_train).astype(np.float32).mean())

    results = model.apply(x_test)
    print("Testing Accuracy: ", np.equal(np.argmax(results, axis=1), y_test).astype(np.float32).mean())

    print(model.apply(np.array([[1,2,3,4,5,6,7,8,9,0], [0,1,0,0,0,0,0,0,0,0], [0,9,0,0,0,0,0,0,0,0]])))


if __name__ == "__main__":
    main()