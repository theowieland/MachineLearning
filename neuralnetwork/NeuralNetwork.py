import abc
import json

import numpy as np
from abc import ABC


class ActivationFunction(ABC):

    @abc.abstractmethod
    def identifier(self):
        pass

    @abc.abstractmethod
    def activate(self, x):
        pass

    @abc.abstractmethod
    def derivative(self, y):
        pass

    @abc.abstractmethod
    def initial_weights_range(self, neurons_in, neurons_out) -> (float, float):
        """
        :return: (min_value, max_value)
        """
        pass


class SigmoidActivation(ActivationFunction):

    def identifier(self):
        return "sigmoid"

    def activate(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, y):
        return [value * (1 - value) for value in y]

    def initial_weights_range(self, neurons_in, neurons_out) -> (float, float):
        divisor = np.sqrt(neurons_in + neurons_out)

        return -1.0 / divisor, 1.0 / divisor


class TanhActivation(ActivationFunction):

    def identifier(self):
        return "tanh"

    def activate(self, x):
        return np.tanh(x)

    def derivative(self, y):
        return 1 - np.power(y, 2)

    def initial_weights_range(self, neurons_in, neurons_out) -> (float, float):
        divisor = np.sqrt(neurons_in + neurons_out)

        return -1.0 / divisor, 1.0 / divisor


class ReLUActivation(ActivationFunction):

    def identifier(self):
        return "relu"

    def activate(self, x):
        return [max(0, value) for value in x]

    def derivative(self, y):
        return [1 if value > 0 else 0 for value in y]

    def initial_weights_range(self, neurons_in, neurons_out) -> (float, float):
        r = np.sqrt(12 / (neurons_in + neurons_out))
        return -r, r


class LeakyReLUActivation(ActivationFunction):

    def identifier(self):
        return "leaky relu"

    def activate(self, x):
        return [value if value > 0 else 0.01 * value for value in x]

    def derivative(self, y):
        return [1 if value > 0 else 0.01 for value in y]

    def initial_weights_range(self, neurons_in, neurons_out) -> (float, float):
        r = np.sqrt(12 / (neurons_in + neurons_out))
        return -r, r


# stores all available activation functions so they can be accessed while importing a neural network from a file
ACTIVATION_FUNCTIONS = [SigmoidActivation(), TanhActivation(), ReLUActivation(), LeakyReLUActivation()]


class NeuralNetwork:

    def __init__(self, layers=[], activation=SigmoidActivation()):
        self.layers = layers
        self.activation = activation

        self.weights = [np.zeros((self.layers[layer], self.layers[layer - 1])) for layer in range(1, len(self.layers))]
        for layer in range(1, len(layers)):
            (range_min, range_max) = activation.initial_weights_range(self.layers[layer - 1], self.layers[layer])
            self.weights[layer - 1] = np.random.uniform(range_min, range_max, size=(self.layers[layer], self.layers[layer - 1]))

        self.biases = [np.zeros(self.layers[layer]) for layer in range(1, len(layers))]

        self.weight_adjustments = [np.zeros((self.layers[layer], self.layers[layer - 1])) for layer in range(1, len(layers))]
        self.bias_adjustments = [np.zeros(self.layers[layer]) for layer in range(1, len(layers))]

        # stores the output of each layer after one forward pass
        self.outputs = [np.zeros(self.layers[layer]) for layer in range(0, len(layers))]

    def output(self):
        return self.outputs[len(self.layers) - 1]

    def classification(self, inputs):
        self.feed_forward(inputs)
        return [round(value) for value in self.output()]

    def feed_forward(self, inputs):
        self.outputs[0] = inputs

        for layer in range(1, len(self.layers)):
            self.outputs[layer] = np.dot(self.weights[layer -1], self.outputs[layer - 1]) + self.biases[layer - 1]

            # apply activation function
            self.outputs[layer] = self.activation.activate(self.outputs[layer])

        return self.output()

    def feed_backward(self, expected_output, learning_rate=0.1):
        previous_layer_error = np.subtract(self.output(), expected_output)

        for layer in range(len(self.layers) - 1, 0, -1):
            deltas = np.multiply(previous_layer_error, self.activation.derivative(self.outputs[layer]))

            self.weight_adjustments[layer - 1] = learning_rate * np.outer(deltas, self.outputs[layer - 1])
            self.bias_adjustments[layer - 1] = learning_rate * deltas
            previous_layer_error = self.weights[layer - 1].T @ deltas

    def update_weights_and_biases(self):
        for layer in range(0, len(self.layers) - 1):
            self.weights[layer] -= self.weight_adjustments[layer]
            self.biases[layer] -= self.bias_adjustments[layer]

    def train(self, dataset, num_iterations, learning_rate=.015, print_debug=True):
        debug_steps = num_iterations / 25

        for iteration in range(num_iterations):
            if (iteration % debug_steps) == 0 and print_debug:
                print("training progress: %d%%" % ((iteration / num_iterations) * 100))

            for (inputs, expected_output) in dataset:
                self.feed_forward(inputs)
                self.feed_backward(expected_output, learning_rate)
                self.update_weights_and_biases()

    def evaluate(self, dataset):
        correct = 0

        for (inputs, expected_output) in dataset:
            self.feed_forward(inputs)
            if np.array_equiv(self.classification(inputs), expected_output):
                correct += 1

        return correct

    def print(self):
        print(self.weights)
        print(self.biases)

    def export_to_file(self, file):
        data = dict()

        data['layers'] = self.layers
        data['activation'] = self.activation.identifier()
        data['weights'] = dict()
        data['biases'] = dict()

        for layer in range(0, len(self.layers) - 1):
            data['weights'][str(layer)] = self.weights[layer].tolist()
            data['biases'][str(layer)] = self.biases[layer].tolist()

        with open(file, 'w') as file:
            json.dump(data, file)

    def import_from_file(self, file):
        with open(file, 'r') as file:
            data = json.load(file)

        self.layers = data['layers']
        self.weights = [np.zeros((self.layers[layer], self.layers[layer - 1])) for layer in range(1, len(self.layers))]
        self.biases = [np.zeros(self.layers[layer]) for layer in range(1, len(self.layers))]
        self.weight_adjustments = [np.zeros((self.layers[layer], self.layers[layer - 1])) for layer in range(1, len(self.layers))]
        self.bias_adjustments = [np.zeros(self.layers[layer]) for layer in range(1, len(self.layers))]
        self.outputs = [np.zeros(self.layers[layer]) for layer in range(0, len(self.layers))]

        for activation in ACTIVATION_FUNCTIONS:
            if activation.identifier() == data['activation']:
                self.activation = activation

        for layer in range(0, len(self.layers) - 1):
            self.weights[layer] = np.array(data['weights'][str(layer)])
            self.biases[layer] = np.array(data['biases'][str(layer)])
