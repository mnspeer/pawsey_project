import numpy as np

from mnist import MNIST


def softmax(x: np.array) -> np.array:
    """Apply softmax independently to each row."""
    z = np.exp(x - x.max(1)[:, None])
    return z / z.sum(1)[:, None]


def main():
    learning_rate = 0.01
    batch_size = 256
    n_epochs = 4

    mnist = MNIST()

    weights = np.random.randn(784, 10) * np.sqrt(2 / 784)
    for _ in range(n_epochs):
        for images, labels in mnist.train_set.minibatches(batch_size):
            preds = softmax(images.dot(weights))
            error = images.T.dot(preds - labels)
            weights -= learning_rate * error

            acc = np.mean(preds.argmax(1) == labels.argmax(1))
            print(f'Train acc: {acc}')


if __name__ == '__main__':
    main()
