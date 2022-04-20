import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(y):
    return y * (1 - y)


class NeuralNetwork:

    def __init__(self, layers):
        self.layers = layers

        self.weights = [np.random.rand(self.layers[layer], self.layers[layer - 1]) for layer in range(1, len(layers))]
        self.biases = [np.random.rand(self.layers[layer]) for layer in range(1, len(layers))]

    def layer_results(self, inputs):
        """
        calculate the output for each layer

        :param inputs: the input vector
        :return: the output vector for each layer
        """
        results = list()
        previous_layer_result = inputs
        results.append(previous_layer_result)

        for layer in range(1, len(self.layers)):
            previous_layer_result = np.dot(self.weights[layer - 1], previous_layer_result) + self.biases[layer - 1]

            # apply activation function
            previous_layer_result = np.array([sigmoid(x) for x in previous_layer_result])

            results.append(previous_layer_result)

        return results

    def feedforward(self, input):
        # return the result vector from the last layer
        return self.layer_results(input)[len(self.layers) - 1]

    def train(self, input, expected_result, learning_rate=.1):
        layer_results = self.layer_results(input)

        previous_layer_error = layer_results[len(self.layers) - 1] - expected_result

        all_weight_adjustments = list()
        all_bias_adjustments = list()

        for layer in range(len(self.layers) - 1, 0, -1):
            layer_bias_adjustments = np.zeros(shape=(self.layers[layer]))
            layer_weight_adjustments = np.zeros(shape=(self.layers[layer], self.layers[layer - 1]))
            activation_function_errors = np.zeros(shape=(self.layers[layer]))

            for current_neuron in range(self.layers[layer]):
                activation_function_errors[current_neuron] = previous_layer_error[current_neuron] * sigmoid_derivative(layer_results[layer][current_neuron])
                layer_bias_adjustments[current_neuron] = activation_function_errors[current_neuron]

                layer_weight_adjustments[current_neuron] = [activation_function_errors[current_neuron] * layer_results[layer - 1][previous_neuron] for previous_neuron in range(self.layers[layer - 1])]

            previous_layer_error = np.zeros(shape=(self.layers[layer - 1]))

            for previous_neuron in range(self.layers[layer - 1]):
                previous_layer_error[previous_neuron] = sum([activation_function_errors[current_neuron] * self.weights[layer - 1][current_neuron][previous_neuron] for current_neuron in range(self.layers[layer])])

            all_weight_adjustments.append(layer_weight_adjustments)
            all_bias_adjustments.append(layer_bias_adjustments)

        all_weight_adjustments.reverse()
        all_bias_adjustments.reverse()

        for layer in range(len(self.layers) - 1):
            self.weights[layer] -= learning_rate * all_weight_adjustments[layer]
            self.biases[layer] -= learning_rate * all_bias_adjustments[layer]

    def print(self):
        print(self.weights)
        print(self.biases)
