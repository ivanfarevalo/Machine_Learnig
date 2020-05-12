import sample_generation
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pprint
import math


class K_means():

    def __init__(self, num_clusters, samples):
        self.num_clusters = num_clusters
        self.samples = samples
        self.one_hot_assignments = []
        self.cluster_means = []
        self.num_iterations = 0
        self.distortion = None
        self.jth_center_probabilities = np.ones(self.samples.shape[0])/self.samples.shape[0]

    def update_cluster_centers(self, initial_assignment=False, k_meansplusplus=False):

        if initial_assignment:  # No seed provided, each run has a random cluster initialization
            np.random.seed()
            while len(self.cluster_means) < self.num_clusters:
                index = np.random.choice(np.arange(0, np.shape(self.samples)[0]), p=self.jth_center_probabilities)
                # index = np.random.randint(0, np.shape(self.samples)[0])  # Generate random index for cluster center
                if not any((self.samples[index]  == cluster_mean).all() for cluster_mean in self.cluster_means):
                    self.cluster_means.append(self.samples[index])
                    if k_meansplusplus and len(self.cluster_means) < self.num_clusters:
                        self.assign_data_points(update_jth_center_probability=True)
        else:
            for i in range(self.num_clusters):
                self.cluster_means[i] = np.mean(self.samples[np.array(self.one_hot_assignments[:, i], dtype=bool)], axis=0)

    def assign_data_points(self, update_jth_center_probability=False):

        distances = np.array([])

        for cluster_mean in self.cluster_means:
            distance_to_cluster = np.square(np.linalg.norm(self.samples - cluster_mean, axis=1))
            # distance_to_cluster = np.square(np.linalg.norm(self.samples - cluster_mean, axis=1))
            if np.any(distances):  # If not empty
                distances = np.c_[distances, distance_to_cluster]  # Append norm of kth cluster to array
            else:
                distances = distance_to_cluster  # If empty, create matrix with norm of first cluster

        self.one_hot_assignments = np.eye(len(self.cluster_means))[np.argmin(distances.reshape(-1, len(self.cluster_means)), axis=1).reshape(-1)]  #  One hot assignment
        self.distortion = sum([np.sum(np.square(np.linalg.norm(self.samples[np.array(self.one_hot_assignments[:,i], dtype=bool)]- self.cluster_means[i], axis=1))) for i in range(len(self.cluster_means))])
        if update_jth_center_probability:
            self.jth_center_probabilities = np.divide(distances[np.array(self.one_hot_assignments, dtype=bool).squeeze()], self.distortion)
        else:
            self.num_iterations += 1

    def print_info(self):
        print("K = {} clusters\nTotal number of iterations: {}\nDistortion: {}\n\n".
              format(self.num_clusters, self.num_iterations, self.distortion))


class Expectation_Maximization():

    def __init__(self, kmean_estimation):
        self.num_clusters = kmean_estimation.num_clusters
        self.samples = kmean_estimation.samples
        self.mixture_means = kmean_estimation.cluster_means
        self.num_iterations = 0
        self.posterior_probabilities = kmean_estimation.one_hot_assignments
        self.kth_effective_points = np.sum(self.posterior_probabilities, axis=0)
        self.priors = self.kth_effective_points / len(self.samples)
        self.mixture_covariences = list(np.zeros([self.num_clusters, self.samples.shape[1], self.samples.shape[1]]))
        self.update_mixture_covariences()
        self.update_cost()

    def update_conditional_probabilities(self):
        self.cond_probabilites = np.zeros(self.posterior_probabilities.shape)
        self.log_cond_probabilites = np.zeros(self.posterior_probabilities.shape)
        for k in range(self.num_clusters):
            gaussian_distribution = multivariate_normal(self.mixture_means[k], self.mixture_covariences[k])
            self.cond_probabilites[:, k] = gaussian_distribution.pdf(self.samples)
            self.log_cond_probabilites[:, k] = gaussian_distribution.logpdf(self.samples)

    def update_posterior_probabilities(self):
        # self.update_conditional_probabilities()
        self.posterior_probabilities = (1/(np.matmul(self.cond_probabilites, self.priors)).reshape([-1, 1]))*(self.cond_probabilites*self.priors)
        self.kth_effective_points = np.sum(self.posterior_probabilities, axis=0)


    def update_priors(self):
        self.priors = self.kth_effective_points / len(self.samples)

    def update_mixture_means(self):
        for k in range(self.num_clusters):
            self.mixture_means[k] = np.sum(self.posterior_probabilities[:,k].reshape([-1,1])*self.samples, axis=0) / self.kth_effective_points[k]

    def update_mixture_covariences(self):
        for k in range(self.num_clusters):
            self.mixture_covariences[k] = np.sum(
                [self.posterior_probabilities[i, k]*np.outer(self.samples[i]-self.mixture_means[k], self.samples[i]-self.mixture_means[k])
                 for i in range(len(self.samples))], axis=0) / self.kth_effective_points[k]

    def update_cost(self):
        self.cost = 0
        self.update_conditional_probabilities()
        for k in range(self.num_clusters):
             for i in range(len(self.samples)):
                 self.cost += self.posterior_probabilities[i, k] * (self.log_cond_probabilites[i, k] + math.log(self.priors[k]))
        self.num_iterations += 1


def generate_data(num_samples=200, plot_data=False):
    seed1, seed2, seed3 = 14, 8, 1228  # Seed for each class
    class_1, _, class_2, class_3 = sample_generation.generate_samples(num_samples, seed0=seed1, seed1A=seed2,
                                                                      seed1B=seed3)
    class_1.prior, class_2.prior, class_3.prior = 1 / 2, 1 / 6, 1 / 3

    np.random.seed(seed1 + seed2 + seed3)
    prior = np.random.rand(num_samples)
    samples = np.zeros([num_samples, 2])
    one_hot_labels = np.zeros([num_samples, 3])
    for i in range(num_samples):
        if prior[i] < class_1.prior:
            samples[i, :] = class_1.data[i, :]  # Matrix with samples
            one_hot_labels[i, :] = np.array([1, 0, 0])  # One hot encoding matrix
        elif prior[i] < class_1.prior + class_2.prior:
            samples[i, :] = class_2.data[i, :]  # Matrix with samples
            one_hot_labels[i, :] = np.array([0, 1, 0])  # One hot encoding matrix
        else:
            samples[i, :] = class_3.data[i, :]  # Matrix with samples
            one_hot_labels[i, :] = np.array([0, 0, 1])  # One hot encoding matrix

    if plot_data:
        plt.figure(1)
        plt.plot(class_1.data[np.equal(class_1.data, samples)[:, 0], 0],
                 class_1.data[np.equal(class_1.data, samples)[:, 1], 1], 'b.')
        plt.plot(class_2.data[np.equal(class_2.data, samples)[:, 0], 0],
                 class_2.data[np.equal(class_2.data, samples)[:, 1], 1], 'r.')
        plt.plot(class_3.data[np.equal(class_3.data, samples)[:, 0], 0],
                 class_3.data[np.equal(class_3.data, samples)[:, 1], 1], 'g.')
        plt.axis('equal')
        plt.legend(['Class 1', 'Class 2', 'Class 3'])

        plt.figure(2)
        plt.plot(samples[:, 0], samples[:, 1], 'k.')
        plt.axis('equal')
        plt.legend(['Samples'])
        # plt.show()
    return samples, one_hot_labels


def plot_probability_tables(probability_matrix):
    for i in range(len(probability_matrix)):
        # Prepare table
        columns = np.arange(1, probability_matrix[i].shape[1] + 1)
        rows = np.arange(1, probability_matrix[i].shape[0] + 1)
        cell_data = probability_matrix[i]
        normal = plt.Normalize(cell_data.min() - 0.3, cell_data.max() + 0.3)
        colors = plt.cm.coolwarm(normal(cell_data))

        F = plt.figure(figsize=(15, 8))
        ax = F.add_subplot(111, frameon=True, xticks=[], yticks=[])
        ax.axis('tight')
        ax.axis('off')
        plt.xlabel("K")
        plt.ylabel("l")
        the_table = ax.table(cellText=cell_data, colWidths=[0.2] * columns.size,
                             colLabels=columns, rowLabels=rows, loc='center', cellColours=colors)
    # plt.show()


def run_Kmeans_algorithm(samples, one_hot_labels, rangeK=np.arange(1, 10), num_trials=100, k_meansplusplus=False, plot_tables=False):

    def plot_distortion():
        # Plot the distortion as a function of K.
        fig = plt.figure()
        distortions = [k_means.distortion for k_means in k_means_list]
        plt.plot(rangeK, distortions, 'ro')
        plt.xlabel("Number of clusters")
        plt.ylabel("Distortion")

        # Plot the ratio as a function of K.
        fig2 = plt.figure()
        distortion_ratios = [k_means_list[i + 1].distortion / k_means_list[i].distortion for i in range(len(k_means_list) - 1)]
        plt.plot(distortion_ratios, 'bo')
        plt.xlabel("Number of clusters")
        plt.ylabel("Distortion Ratio (K_i+1)/(K_i)")
        # plt.show()

    # K-means Algorithm
    jth_k_mean = 0
    epsilon = 0.000001
    k_means_list = []
    probability_matrix = []
    for num_clusters in rangeK:  # For K = 2, 3, 4, 5

        for trial in range(num_trials):  # Get best estimate from num_trials

            k_means_trial = K_means(num_clusters, samples)
            k_means_trial.update_cluster_centers(initial_assignment=True, k_meansplusplus=k_meansplusplus)  # Change for k-means++
            k_means_trial.assign_data_points()
            while k_means_trial.num_iterations < 20:
                previous_means = k_means_trial.cluster_means.copy()
                k_means_trial.update_cluster_centers()
                k_means_trial.assign_data_points()
                if np.all(abs(np.subtract(k_means_trial.cluster_means, previous_means)) < epsilon):
                    break
            # k_means_trial.print_info()
            try:
                if k_means_trial.distortion < k_means_list[jth_k_mean].distortion:  # Append trial with least distortion
                    k_means_list[jth_k_mean] = k_means_trial
            except IndexError:
                k_means_list.append(k_means_trial)

        print("Best Iteration:")
        k_means_list[jth_k_mean].print_info()

        L = np.count_nonzero(one_hot_labels, axis=0)  # Number of samples from each class L
        kth_probability_matrix = []
        for l in range(one_hot_labels.shape[1]):  # For each mixture component
            ith_label_indices = np.nonzero(one_hot_labels[:, l])[0]  # Get sample indices generate from lth component
            K = np.array(
                [np.count_nonzero(k_means_list[jth_k_mean].one_hot_assignments[ith_label_indices, k], axis=0) for k in
                 range(num_clusters)]).squeeze()  # Number of samples assigned to K that came from L
            kth_probability_matrix.append(np.divide(K, L[l]))

        probability_matrix.append(np.array(kth_probability_matrix))
        jth_k_mean += 1

    # Plot empirical probability tables
    if plot_tables:
        plot_probability_tables(probability_matrix)  # Generates plot figures
        plot_distortion()  # Plot the distortion as a function of K.
    for i in range(len(probability_matrix)): print("{}K = {}\n{}\n".format((2 * i + 2) * "    ", i + 2, probability_matrix[i]))




    return k_means_list


def run_EM_algorithm(kmeans_tests, one_hot_labels,  plot_tables=False, epsilon=0.0001):
    jth_estimation = 0
    gaussian_mixture_list = []
    probability_matrix = []
    for kmeans_test in kmeans_tests:
        mixture_estimation = Expectation_Maximization(kmeans_test)
        while mixture_estimation.num_iterations < 200:
            mixture_estimation.update_posterior_probabilities()
            mixture_estimation.update_priors()
            mixture_estimation.update_mixture_means()
            mixture_estimation.update_mixture_covariences()
            previous_cost = mixture_estimation.cost
            mixture_estimation.update_cost()
            print("Iteration : {}\nCost: {}\n".format(mixture_estimation.num_iterations, mixture_estimation.cost))
            if abs(mixture_estimation.cost-previous_cost) < epsilon:
                break
        gaussian_mixture_list.append(mixture_estimation)
        # pprint.pprint(mixture_estimation.__dict__)
        pprint.pprint(mixture_estimation.num_iterations)
        pprint.pprint(mixture_estimation.cost)

        L = np.count_nonzero(one_hot_labels, axis=0)  # Number of samples from each class L
        kth_probability_matrix = []
        for l in range(one_hot_labels.shape[1]):
            ith_label_indices = np.nonzero(one_hot_labels[:, l])[0]  # Get sample indices generate from lth component
            K = np.array(
                [np.sum(gaussian_mixture_list[jth_estimation].posterior_probabilities[ith_label_indices, k], axis=0) for k in
                 range(gaussian_mixture_list[jth_estimation].num_clusters)]).squeeze()  # Number of samples assigned to K that came from L

            kth_probability_matrix.append(np.divide(K, L[l]))

        probability_matrix.append(np.array(kth_probability_matrix))
        jth_estimation += 1

    # Plot empirical probability tables
    if plot_tables: plot_probability_tables(probability_matrix)  # Generates plot figures
    for i in range(len(probability_matrix)): print("{}K = {}\n{}\n".format((2 * i + 2) * "    ", i + 2, probability_matrix[i]))


def find_quasi_orthogonal_vectors(dim, num_vectors, from_memory=False):
    if from_memory:
        return sample_generation.generate_d_dim_samples(dim, num_vectors, best_seeds=[3770, 350, 86, 861, 5616, 2738])
    else:
        cost_list = []
        best_seeds = [3770]
        for j in range(2, num_vectors + 1):
            for i in range(10000):
                trial_seed = list(best_seeds)
                trial_seed.append(i)
                q_orthoganal_vectors = sample_generation.generate_d_dim_samples(dim, j, best_seeds=trial_seed)
                normalized_vectors = q_orthoganal_vectors / np.linalg.norm(q_orthoganal_vectors, axis=1).reshape([-1, 1])
                check_orthogonality = normalized_vectors.dot(normalized_vectors.T)
                cost = np.sum(abs(check_orthogonality - np.diag(check_orthogonality.diagonal())))
                if np.count_nonzero(q_orthoganal_vectors[j-1]) in range(8, 13):
                    cost_list.append(cost)
                else:
                    cost_list.append(100)
            print("Best cost: {}\nBest seed: {}\n".format(min(cost_list), cost_list.index(min(cost_list))))
            best_seeds.append(cost_list.index(min(cost_list)))
            cost_list.clear()
        return sample_generation.generate_d_dim_samples(dim, num_vectors, best_seeds=best_seeds)


def generate_d_dim_data(basis, num_samples=50):
    #  Create V1, V2, choice, and Noise vectors
    np.random.seed(785)
    v1 = np.random.normal(size=num_samples)
    np.random.seed(3589)
    v2 = np.random.normal(size=num_samples)
    random_state = np.random.RandomState(13489)
    noise = random_state.multivariate_normal(np.zeros(len(basis[0])), 0.01*np.eye(len(basis[0])), num_samples)
    np.random.seed(89)
    choice = np.random.choice([1,2,3], p=[1/3, 1/3, 1/3], size=num_samples)
    samples = np.zeros([num_samples, len(basis[0])])
    one_hot_labels = np.zeros([num_samples, 3])
    # samples = []
    # one_hot_labels = []
    for i in range(num_samples):
        if choice[i] == 1:
            samples[i, :] = basis[0] + v1[i]*basis[1] + v2[i]*basis[2] + noise[i]
            one_hot_labels[i, :] = np.array([1, 0, 0])
        elif choice[i] == 2:
            samples[i, :] = 2*basis[3] + math.sqrt(2)*v1[i]*basis[4] + v2[i]*basis[5] + noise[i]
            one_hot_labels[i, :] = np.array([0, 1, 0])
        else:
            samples[i, :] = math.sqrt(2)*basis[5] + v1[i]*(basis[0] + basis[1]) + 1/math.sqrt(2)*v2[i]*basis[4] + noise[i]
            one_hot_labels[i, :] = np.array([0, 0, 1])
    return samples, one_hot_labels


if __name__ == '__main__':

    # Generate Samples
    samples, one_hot_labels = generate_data(num_samples=200, plot_data=False)
    # Run Kmeans algorithm
    run_Kmeans_algorithm(samples, one_hot_labels, rangeK=np.arange(2, 10), plot_tables=True)  # Can only plot for K > 1
    # Run Kmeans++ algorithm
    run_Kmeans_algorithm(samples, one_hot_labels, rangeK=np.arange(2, 10), k_meansplusplus=True, plot_tables=False)
    # Show plots and tables.
    plt.show()

    # Run Expectation Maximization algorithm
    kmeans_tests = run_Kmeans_algorithm(samples, one_hot_labels, rangeK=np.arange(3, 4), k_meansplusplus=True, plot_tables=False)
    run_EM_algorithm(kmeans_tests, one_hot_labels, plot_tables=True)
    plt.show()

    # Generate quasi-orthogonal matrices
    q_orthoganal_vectors = find_quasi_orthogonal_vectors(30, 6, from_memory=True)
    # Generate D-dim samples from basis
    d_dim_samples, one_hot_labels = generate_d_dim_data(q_orthoganal_vectors, num_samples=250)

    # Run K-Means algorithm with K_means++ initialization
    d_dim_kmeans_tests = run_Kmeans_algorithm(d_dim_samples, one_hot_labels, rangeK=np.arange(3, 4), k_meansplusplus=True, plot_tables=False)
    run_EM_algorithm(d_dim_kmeans_tests, one_hot_labels, plot_tables=True, epsilon=0.001)
    # Show plots and tables.
    plt.show()

    pass

