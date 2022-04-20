import numpy as np
import time

from neuralnetwork.NeuralNetwork import NeuralNetwork

if __name__ == '__main__':
    nn = NeuralNetwork([2, 2, 1])

    start_time = time.time()

    for iteration in range(0, 10000):
        nn.train([1, 0], [1])
        nn.train([0, 1], [1])
        nn.train([1, 1], [1])
        nn.train([0, 0], [0])

    print("--- %s seconds ---" % (time.time() - start_time))

    print("feed forward result: " + str(nn.feedforward(np.asarray([2, 2]))))
    print("feed forward result: " + str(nn.feedforward(np.asarray([0, 0]))))

