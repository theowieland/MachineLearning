from neuralnetwork.NeuralNetwork import NeuralNetwork

if __name__ == '__main__':
    nn = NeuralNetwork([3, 2, 1])
    nn.print()
    print("feed forward result: " + str(nn.feedforward([1, 2, 3])))
