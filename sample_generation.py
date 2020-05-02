import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal


class Gaussian:

    def __init__(self, mean=None, covariance=None, eigenvalues=None, eigenvectors=None):
        self.mean = mean
        self.covariance = covariance
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.data = None

    def get_covariance(self):
        self.covariance = self.eigenvectors @ self.eigenvalues @ self.eigenvectors.T

    def generate_data(self, number_of_samples, seed=None):
        if self.covariance is None:
            self.get_covariance()
        random_state = np.random.RandomState(seed)
        self.data = random_state.multivariate_normal(self.mean, self.covariance, number_of_samples)
        # self.data = np.random.multivariate_normal(self.mean, self.covariance, number_of_samples, random_state=self.seed)


def generate_samples(num_samples, seed0=None, seed1A=None, seed1B=None):
    # Class 0
    class_0 = Gaussian()
    class_0.mean = 1*np.array([1, -1])
    class_0.eigenvectors = np.eye(2)
    class_0.eigenvalues = np.array([[1, 0], [0, 4]])
    class_0.generate_data(num_samples, seed0)

    # Class 1A
    class_1A = Gaussian()
    class_1A.prior = 2 / 3
    class_1A.theta = -3 * np.pi / 4
    class_1A.mean = np.array([-1, 0])
    class_1A.eigenvectors = np.array([[math.cos(class_1A.theta), -1 * math.sin(class_1A.theta)],
                                      [math.sin(class_1A.theta), math.cos(class_1A.theta)]])
    class_1A.eigenvalues = np.array([[4, 0], [0, 0.5]])
    class_1A.get_covariance()

    # Class 1B
    class_1B = Gaussian()
    class_1B.prior = 1 / 3
    class_1B.theta = np.pi / 4
    class_1B.mean = np.array([4, 1])
    class_1B.eigenvectors = np.array([[math.cos(class_1B.theta), -1 * math.sin(class_1B.theta)],
                                      [math.sin(class_1B.theta), math.cos(class_1B.theta)]])
    class_1B.eigenvalues = np.array([[1, 0], [0, 4]])
    class_1B.get_covariance()

    # Class 1
    np.random.seed(seed1A + seed1B)
    prior = np.random.rand(num_samples)
    class_1 = np.zeros([num_samples, 2])
    for i in range(num_samples):
        if prior[i] > class_1B.prior:
            class_1A.generate_data(1, seed1A+i)
            class_1[i, :] = class_1A.data
        else:
            class_1B.generate_data(1, seed1B+i)
            class_1[i, :] = class_1B.data

    # Return Samples
    return class_0, class_1, class_1A, class_1B


if __name__ == '__main__':

    # HW 1 PART 1

    num_samples = 200  # For each class
    class_0, class_1, class_1A, class_1B = generate_samples(num_samples, seed0=145, seed1A=4598, seed1B=6548)
    plt.plot(class_0.data[:, 0], class_0.data[:, 1], '.')
    plt.plot(class_1[:, 0], class_1[:, 1], 'r.')
    plt.axis('equal')
    plt.legend(['Class 0', 'Class 1'])
    plt.show()
