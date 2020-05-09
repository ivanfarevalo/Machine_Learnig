import sample_generation
import numpy as np
import matplotlib.pyplot as plt


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

        if initial_assignment:
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


def plot_probability_tables(probability_matrix):
    for i in range(len(probability_matrix)):
        # Prepare table
        columns = np.arange(1, probability_matrix[i].shape[1] + 1)
        rows = np.arange(1, probability_matrix[i].shape[0] + 1)
        cell_data = probability_matrix[i]
        normal = plt.Normalize(cell_data.min() -0.3, cell_data.max() + 0.3)
        colors = plt.cm.coolwarm(normal(cell_data))

        F = plt.figure(figsize=(15,8))
        ax = F.add_subplot(111, frameon=True, xticks=[], yticks=[])
        ax.axis('tight')
        ax.axis('off')
        plt.xlabel("K")
        plt.ylabel("l")
        the_table = ax.table(cellText=cell_data, colWidths = [0.2]*cell_data.shape[1],
                             colLabels=columns, rowLabels=rows,loc='center', cellColours=colors)
        plt.show()


if __name__ == '__main__':

    # Generate Samples
    num_samples = 200  # Total number of samples
    seed1, seed2, seed3 = 145, 48, 18  # Seed for each class
    class_1, _, class_2, class_3 = sample_generation.generate_samples(num_samples, seed0=seed1, seed1A=seed2, seed1B=seed3)
    class_1.prior, class_2.prior, class_3.prior = 1/2, 1/6, 1/3

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

    plt.figure(1)
    plt.plot(class_1.data[np.equal(class_1.data, samples)[:,0], 0], class_1.data[np.equal(class_1.data, samples)[:, 1], 1], 'b.')
    plt.plot(class_2.data[np.equal(class_2.data, samples)[:,0], 0], class_2.data[np.equal(class_2.data, samples)[:, 1], 1], 'r.')
    plt.plot(class_3.data[np.equal(class_3.data, samples)[:,0], 0], class_3.data[np.equal(class_3.data, samples)[:, 1], 1], 'g.')
    plt.axis('equal')
    plt.legend(['Class 1', 'Class 2', 'Class 3'])

    plt.figure(2)
    plt.plot(samples[:, 0], samples[:, 1], 'k.')
    plt.axis('equal')
    plt.legend(['Samples'])
    plt.close('all')
    # plt.show()

    # Implement K-means algorithm
    jth_k_mean = 0
    num_trials = 10
    epsilon = 0.000001
    k_means_list = []
    probability_matrix = []
    rangeK = np.arange(2, 6)
    for num_clusters in rangeK:  # For K = 2, 3, 4, 5

        for trial in range(num_trials):  # Get best estimate from num_trials

            k_means_trial = K_means(num_clusters, samples)
            k_means_trial.update_cluster_centers(initial_assignment=True, k_meansplusplus=True)
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

        L = np.count_nonzero(one_hot_labels, axis=0)  # Number of samples from each class
        kth_probability_matrix = []
        for l in range(one_hot_labels.shape[1]):
            ith_label_indices = np.nonzero(one_hot_labels[:, l])[0]
            K = np.array([np.count_nonzero(k_means_list[jth_k_mean].one_hot_assignments[ith_label_indices, k], axis=0) for k in range(num_clusters)]).squeeze()
            kth_probability_matrix.append(np.divide(K, L[l]))

        probability_matrix.append(np.array(kth_probability_matrix))
        jth_k_mean += 1

    # Plot empirical probability tables
    # plot_probability_tables(probability_matrix)
    for i in range(len(probability_matrix)): print("{}K = {}\n{}\n".format((2*i+2)*"    ",i+2, probability_matrix[i]))

    # Plot the distortion as a function of K.
    fig = plt.figure()
    distortions = [k_means.distortion for k_means in k_means_list]
    plt.plot(rangeK, distortions, 'ro')
    plt.xlabel("Number of clusters")
    plt.ylabel("Distortion")
    plt.show()

    # Implement K-means++ algorithm


