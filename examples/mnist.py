import gzip
import numpy as np

from neuralnetwork.NeuralNetwork import NeuralNetwork, ReLUActivation

FOLDER = ""

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

TRAINING_SIZE = 60000
TRAINING_IMAGES = "train-images-idx3-ubyte.gz"
TRAINING_LABELS = "train-labels-idx1-ubyte.gz"

TEST_SIZE = 10000
TEST_IMAGES = "t10k-images-idx3-ubyte.gz"
TEST_LABELS = "t10k-labels-idx1-ubyte.gz"


def read_images(path, num_images, image_width, image_height):
    file = gzip.open(path, 'r')
    file.read(16)

    buffer = file.read(num_images * image_width * image_height)
    data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_width, image_height, 1)

    return data


def read_labels(path, num_labels):
    f = gzip.open(path, 'r')
    f.read(8)

    buffer = f.read(num_labels)
    labels = np.frombuffer(buffer, dtype=np.uint8).astype(np.int64)

    return labels


def normalize_vec(vec, max_value):
    return np.divide(vec, max_value)


if __name__ == '__main__':
    training_images = read_images(FOLDER + "/" + TRAINING_IMAGES, TRAINING_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT)
    training_labels = read_labels(FOLDER + "/" + TRAINING_LABELS, TRAINING_SIZE)

    test_images = read_images(FOLDER + "/" + TEST_IMAGES, TEST_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT)
    test_labels = read_labels(FOLDER + "/" + TEST_LABELS, TEST_SIZE)

    nn = NeuralNetwork([IMAGE_WIDTH * IMAGE_HEIGHT, 16, 16, 10], activation=ReLUActivation())

    # prepare training set
    training_set = list()
    for training_image_index in range(1):
        expected_result = np.zeros(10)
        expected_result[training_labels[training_image_index]] = 1
        training_set.append((normalize_vec(training_images[training_image_index].flatten(), 255), expected_result))

    # prepare test set
    test_set = list()
    for test_image_index in range(TEST_SIZE):
        expected_result = np.zeros(10)
        expected_result[test_labels[test_image_index]] = 1
        test_set.append((normalize_vec(test_images[test_image_index].flatten(), 255), expected_result))

    nn.train(training_set, 50000, 0.15)
    print("evaluation result: " + str(nn.evaluate(test_set)))
