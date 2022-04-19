import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(y):
    return y * (1 - y)


class NeuralNetwork:

    def __init__(self, layers):
        self.layers = layers

        self.weights = [np.random.rand(self.layers[layer_index], self.layers[layer_index - 1]) for layer_index in range(1, len(layers))]
        self.biases = [np.random.rand(self.layers[layer_index]) for layer_index in range(1, len(layers))]

    def layer_results(self, input):
        """
        calculate the output for each layer

        :param input: the input vector
        :return: the output vector for each layer
        """
        results = [None for _layer_index in range(0, len(self.layers))]
        results[0] = input

        for layer_index in range(1, len(self.layers)):
            results[layer_index] = self.weights[layer_index - 1].dot(results[layer_index - 1]) + self.biases[layer_index - 1]

            # apply activation function
            results[layer_index] = [sigmoid(x) for x in results[layer_index]]

        return results

    def feedforward(self, input):
        # return the result vector from the last layer
        return self.layer_results(input)[len(self.layers) - 1]

    def train(self, input, expected_result):
        pass

    def print(self):
        print(self.weights)
        print(self.biases)
