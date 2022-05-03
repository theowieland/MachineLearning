import gzip
from os import listdir
from os.path import isfile, join

import numpy as np
from PIL import Image

from neuralnetwork.NeuralNetwork import NeuralNetwork, SigmoidActivation, ReLUActivation, TanhActivation, \
    LeakyReLUActivation

FOLDER = ""

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

TRAINING_SIZE = 60000
TRAINING_IMAGES = "train-images-idx3-ubyte.gz"
TRAINING_LABELS = "train-labels-idx1-ubyte.gz"

TEST_SIZE = 10000
TEST_IMAGES = "t10k-images-idx3-ubyte.gz"
TEST_LABELS = "t10k-labels-idx1-ubyte.gz"

EXAMPLE_FOLDER = "mnist_test_images"


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


def read_test_images(path):
    images = [(file, join(path, file)) for file in listdir(path) if isfile(join(path, file))]
    images = [(file_name, Image.open(image)) for (file_name, image) in images]
    images = [(file_name, rgb2gray(np.array(image.getdata()))) for (file_name, image) in images]

    return images


def normalize_vec(vec, max_value):
    return np.divide(vec, max_value)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


if __name__ == '__main__':
    training_images = read_images(FOLDER + "/" + TRAINING_IMAGES, TRAINING_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT)
    training_labels = read_labels(FOLDER + "/" + TRAINING_LABELS, TRAINING_SIZE)

    test_images = read_images(FOLDER + "/" + TEST_IMAGES, TEST_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT)
    test_labels = read_labels(FOLDER + "/" + TEST_LABELS, TEST_SIZE)

    nn = NeuralNetwork([IMAGE_WIDTH * IMAGE_HEIGHT, 128, 16, 10], activation=TanhActivation())

    # prepare training set
    training_set = list()
    for training_image_index in range(TRAINING_SIZE):
        expected_result = np.zeros(10)
        expected_result[training_labels[training_image_index]] = 1
        training_set.append((normalize_vec(training_images[training_image_index].flatten(), 255), expected_result))

    # prepare test set
    test_set = list()
    for test_image_index in range(TEST_SIZE):
        expected_result = np.zeros(10)
        expected_result[test_labels[test_image_index]] = 1
        test_set.append((normalize_vec(test_images[test_image_index].flatten(), 255), expected_result))

    for iteration in range(0, 2000):
        batch = np.random.randint(0, len(training_set), size=128)
        batch_data = [training_set[index] for index in batch]

        nn.train(batch_data, 1, 0.15, print_debug=False)

        evaluation = np.random.randint(0, len(test_set), size=25)
        evaluation_data = [test_set[index] for index in evaluation]
        print(str(iteration) + ": evaluation result (correct): " + str(nn.evaluate(evaluation_data) / len(evaluation_data)))

    print("final evaluation result: " + str(nn.evaluate(test_set)))

    test_images = read_test_images(EXAMPLE_FOLDER)
    for (file_name, image) in test_images:
        print("result: (" + str(file_name) + ") " + str(np.argmax(nn.feed_forward(normalize_vec(image, 255)))))

    nn.print()
