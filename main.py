import numpy as np
import time

from neuralnetwork.NeuralNetwork import NeuralNetwork

if __name__ == '__main__':
    nn = NeuralNetwork([2, 30, 2])

    print("feed forward result: " + str(nn.feed_forward(np.asarray([0, 0]))))
    print("feed forward result: " + str(nn.feed_forward(np.asarray([0, 1]))))
    print("feed forward result: " + str(nn.feed_forward(np.asarray([1, 0]))))
    print("feed forward result: " + str(nn.feed_forward(np.asarray([1, 1]))))

    start_time = time.time()

    x_or_data = list()
    x_or_data.append(([0, 0], [0]))
    x_or_data.append(([1, 0], [1]))
    x_or_data.append(([0, 1], [1]))
    x_or_data.append(([1, 1], [0]))

    nn.train(x_or_data, 10000)

    print("--- %s seconds ---" % (time.time() - start_time))

    print(nn.evaluate(x_or_data))

    print("feed forward result: " + str(nn.feed_forward(np.asarray([0, 0]))))
    print("feed forward result: " + str(nn.feed_forward(np.asarray([0, 1]))))
    print("feed forward result: " + str(nn.feed_forward(np.asarray([1, 0]))))
    print("feed forward result: " + str(nn.feed_forward(np.asarray([1, 1]))))

