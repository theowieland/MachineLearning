import abc

import numpy as np
from abc import ABC


class ActivationFunction(ABC):

    @abc.abstractmethod
    def activate(self, x):
        pass

    @abc.abstractmethod
    def derivative(self, y):
        pass


class SigmoidActivation(ActivationFunction):

    def activate(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, y):
        return y * (1 - y)


class ReLUActivation(ActivationFunction):

    def activate(self, x):
        return max(0, x)

    def derivative(self, y):
        return 1 if y > 0 else 0


class NeuralNetwork:

    def __init__(self, layers, activation=SigmoidActivation()):
        self.layers = layers
        self.activation = activation

        self.weights = [np.random.rand(self.layers[layer], self.layers[layer - 1]) for layer in range(1, len(layers))]
        self.biases = [np.zeros(self.layers[layer]) for layer in range(1, len(layers))]

        self.weight_adjustments = [np.random.rand(self.layers[layer], self.layers[layer - 1]) for layer in range(1, len(layers))]
        self.bias_adjustments = [np.ones(self.layers[layer]) for layer in range(1, len(layers))]

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
            self.outputs[layer] = np.array([self.activation.activate(value) for value in self.outputs[layer]])

        return self.output()

    def feed_backward(self, expected_output, learning_rate=0.1):
        previous_layer_error = np.subtract(self.output(), expected_output)
        for layer in range(len(self.layers) - 1, 0, -1):
            deltas = np.multiply(previous_layer_error, [self.activation.derivative(y) for y in self.outputs[layer]])

            self.weight_adjustments[layer - 1] = learning_rate * np.outer(deltas, self.outputs[layer - 1])
            self.bias_adjustments[layer - 1] = learning_rate * deltas

            previous_layer_error = [
                sum([deltas[current_neuron] * self.weights[layer - 1][current_neuron][
                    previous_neuron]
                     for current_neuron in range(self.layers[layer])])
                for previous_neuron in range(self.layers[layer - 1])]

    def update_weights_and_biases(self):
        for layer in range(0, len(self.layers) - 1):
            self.weights[layer] = np.subtract(self.weights[layer], self.weight_adjustments[layer])
            self.biases[layer] = np.subtract(self.biases[layer], self.bias_adjustments[layer])

    def train(self, dataset, num_iterations, learning_rate=.15):
        for iteration in range(num_iterations):

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

        return correct / len(dataset)

    def print(self):
        print(self.weights)
        print(self.biases)
