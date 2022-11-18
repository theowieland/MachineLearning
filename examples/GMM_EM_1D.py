import numpy as np
import matplotlib.pyplot as plt


def log_likelihood(data, num_clusters, pi, mu, sigma):
    log_likelihood = 0

    for value in data:
        for cluster in range(num_clusters):
            log_likelihood += pi[cluster] * normal_distribution(value, mu[cluster], sigma[cluster])

    return log_likelihood


def normal_distribution(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def calculate_expectation(data, num_clusters, pi, mu, sigma):
    expectation = np.zeros(shape=(len(data), num_clusters))

    for value_index, value in enumerate(data):
        for cluster in range(num_clusters):
            expectation[value_index, cluster] = pi[cluster] * normal_distribution(value, mu[cluster], sigma[cluster])

    expectation /= expectation.sum(axis=1)[:, None]

    return expectation


def calculate_maximization(expectation, data, num_clusters, pi, mu, sigma):
    new_pi = np.zeros(pi.shape)
    new_mu = np.zeros(mu.shape)
    new_sigma = np.zeros(sigma.shape)

    for cluster in range(num_clusters):
        new_pi[cluster] = sum(expectation[:, cluster]) / len(data)
        new_sigma[cluster] = np.sqrt(sum(np.multiply(expectation[:, cluster], np.power(data - mu[cluster], 2))) / sum(expectation[:, cluster]))
        new_mu[cluster] = sum(np.multiply(expectation[:, cluster], data)) / sum(expectation[:, cluster])

    return new_pi, new_mu, new_sigma


if __name__ == "__main__":
    num_clusters = 3

    # initialize parameters of the model
    mu = np.random.uniform(-10, 10, size=num_clusters)  # parameter of the normal distribution
    sigma = np.random.random(size=num_clusters) + 1  # parameter of the normal distribution

    pi = np.random.random(size=num_clusters)  # categorical probability over the clusters
    pi /= sum(pi)

    cluster_1_data = np.random.normal(loc=5, scale=1, size=111)
    cluster_2_data = np.random.normal(loc=0, scale=1, size=44)
    cluster_3_data = np.random.normal(loc=-5, scale=1, size=22)

    sample_data = np.concatenate((cluster_1_data, cluster_2_data, cluster_3_data))

    for iteration in range(1000):
        print("iteration: " + str(iteration) + " log_likelihood: " + str(log_likelihood(sample_data, num_clusters, pi, mu, sigma)))
        expectation = calculate_expectation(sample_data, num_clusters, pi, mu, sigma)

        pi, mu, sigma = calculate_maximization(expectation, sample_data, num_clusters, pi, mu, sigma)

    plt.scatter(x=cluster_1_data, y=[0 for _index in range(len(cluster_1_data))], c='red')
    plt.scatter(x=cluster_2_data, y=[0 for _index in range(len(cluster_2_data))], c='green')
    plt.scatter(x=cluster_3_data, y=[0 for _index in range(len(cluster_3_data))], c='blue')

    xs = np.linspace(-10, 10, 200)

    for cluster in range(num_clusters):
        plt.plot(xs, [pi[cluster] * normal_distribution(x, mu[cluster], sigma[cluster]) for x in xs])

    plt.plot(xs, [sum([pi[cluster] * normal_distribution(x, mu[cluster], sigma[cluster]) for cluster in range(num_clusters)]) for x in xs])

    plt.show()


