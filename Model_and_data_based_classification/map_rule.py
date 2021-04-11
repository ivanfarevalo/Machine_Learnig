import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import math
import sample_generation

from scipy.stats import multivariate_normal


def display_decision_boundary(class_0, class_1, class_1A, class_1B):
    num_pts = 200
    limit = 10
    X = Y  = np.linspace(-limit , limit, num_pts)
    XX,  YY = np.meshgrid(X, Y)
    #
    map_decision = np.zeros([num_pts, num_pts])

    class_0_gaussian = multivariate_normal(class_0.mean, class_0.covariance)  # Create Class 0 Gaussian object
    class_1A_gaussian = multivariate_normal(class_1A.mean, class_1A.covariance)  # Create Class 1A Gaussian object
    class_1B_gaussian = multivariate_normal(class_1B.mean, class_1B.covariance)  # Create Class 1B Gaussian object

    # Evaluate the probability density of the whole grid given each class
    class_0_pdf = np.array([class_0_gaussian.pdf([x,y]) for x, y in zip(np.ravel(XX), np.ravel(YY))]).reshape(XX.shape)
    class_1A_pdf = np.array([class_1A_gaussian.pdf([x,y]) for x, y in zip(np.ravel(XX), np.ravel(YY))]).reshape(XX.shape)
    class_1B_pdf = np.array([class_1B_gaussian.pdf([x,y]) for x, y in zip(np.ravel(XX), np.ravel(YY))]).reshape(XX.shape)
    map_decision[np.add(class_1A_pdf * class_1A.prior, class_1B_pdf * class_1B.prior) > class_0_pdf] = 1

    plt.figure(3)
    plt.contourf(XX, YY, class_0_pdf, alpha=0.7, cmap=plt.cm.magma, antialiased = True)
    plt.contourf(XX, YY, class_1A_pdf, alpha=0.5, cmap=plt.cm.plasma, antialiased = True)
    plt.contourf(XX, YY, class_1B_pdf, alpha=0.3, cmap=plt.cm.cividis, antialiased = True)
    plt.title('Probability density functions for class 0, 1A, and 1B')

    # fig = plt.figure(4)
    # ax = fig.gca(projection='3d')
    # combined_pdf = np.maximum(np.maximum(class_0_pdf, class_1A_pdf), class_1B_pdf)
    # surf = ax.plot_surface(XX, YY, combined_pdf, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    return map_decision, XX, YY

def evaluate_map_rule(num_samples, class_0, class_1, class_1A, class_1B):
    samples = np.r_[class_0.data, class_1.data]  # Concatenate class 0 and 1 samples to calculate map performance
    data_labels = np.r_[np.zeros((num_samples, 1), dtype=int), np.ones((num_samples, 1), dtype=int)]
    map_decision = np.zeros((2*num_samples, 1), dtype=int)

    # Compute MAP Decisions
    class_0_gaussian = multivariate_normal(class_0.mean, class_0.covariance)  # Create Class 0 Gaussian object
    class_1A_gaussian = multivariate_normal(class_1A.mean, class_1A.covariance)  # Create Class 1A Gaussian object
    class_1B_gaussian = multivariate_normal(class_1B.mean, class_1B.covariance)  # Create Class 1B Gaussian object

    class_0_pdf = class_0_gaussian.pdf(samples)  # Evaluate the probability density given class 0 for all samples
    class_1A_pdf = class_1A_gaussian.pdf(samples)  # Evaluate the probability density given class 1A for all samples
    class_1B_pdf = class_1B_gaussian.pdf(samples)  # Evaluate the probability density given class 1B for all samples
    # Compare probability densities for class 0 and 1 and decide class 1 for elements with pdf given class 1 higher than
    #  pdf given class 0
    map_decision[np.add(class_1A_pdf*class_1A.prior, class_1B_pdf*class_1B.prior) > class_0_pdf] = 1

    # Compute error probabilities for each class
    error_probability_class_0 = np.mean(np.not_equal(data_labels[:num_samples], map_decision[:num_samples]))
    error_probability_class_1 = np.mean(np.not_equal(data_labels[num_samples:], map_decision[num_samples:]))

    return samples, map_decision, data_labels, error_probability_class_0, error_probability_class_1


if __name__ == '__main__':
    num_samples = 1000 # For each class
    class_0, class_1, class_1A, class_1B = \
        sample_generation.generate_samples(num_samples, seed0=255, seed1A=400, seed1B=545)
    samples, map_decision, data_labels, error_probability_class_0, error_probability_class_1 = \
        evaluate_map_rule(num_samples, class_0, class_1, class_1A, class_1B)

    map_decision_contour, XX, YY = display_decision_boundary(class_0, class_1, class_1A, class_1B)

    plt.figure(1)
    plt.plot(class_0.data[:, 0], class_0.data[:, 1], '.')
    plt.plot(class_1[:, 0], class_1[:, 1], 'r.')
    plt.axis('equal')
    plt.legend(['Class 0', 'Class 1'])
    plt.title('Random samples form Class 0 and 1')

    plt.figure(2)
    plt.plot(class_0.data[:, 0], class_0.data[:, 1], '.')
    plt.plot(class_1[:, 0], class_1[:, 1], 'r.')
    plt.contour(XX, YY, map_decision_contour, alpha=1, cmap=plt.cm.coolwarm, antialiased=True)
    plt.contourf(XX, YY, map_decision_contour, alpha=.15, cmap=plt.cm.coolwarm, antialiased=True)
    plt.title('Map decision rule and boundaries')

    decision_errors0 = np.not_equal(data_labels[:num_samples, :], map_decision[:num_samples, :])
    decision_errors1 = np.not_equal(data_labels[num_samples:, :], map_decision[num_samples:, :])
    class_0_errors = samples[np.flatnonzero(decision_errors0)]
    class_1_errors = samples[np.flatnonzero(decision_errors1)+num_samples] # Foward index to second half of samples
    plt.plot(class_0_errors[:, 0], class_0_errors[:, 1], 'y+')
    plt.plot(class_1_errors[:, 0], class_1_errors[:, 1], 'k+')
    plt.axis('equal')
    plt.legend(['Class 0 Samples', 'Class 1 Samples', 'Class 0 Map Decision Errors', 'Class 1 Map Decision Errors'])

    print("Class 0 probability error: ", error_probability_class_0)
    print("Class 1 probability error: ", error_probability_class_1)

    plt.show()

