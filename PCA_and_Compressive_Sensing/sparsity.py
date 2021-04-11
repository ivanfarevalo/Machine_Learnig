import Clustering.clustering as ul
import matplotlib.pyplot as plt
import sample_generation
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge


import pprint
import math


def plot_s_wrt_N(q_orthoganal_basis):
    s_list = []
    for i in range(2 * 100, 100 * 100, 1000):
        # Generate D-dim samples from basis
        d_dim_samples, one_hot_labels = ul.generate_d_dim_data(q_orthoganal_basis, num_samples=i)
        U, s, Vh = svd(d_dim_samples-d_dim_samples.mean(0))
        s_list.append(s)
    s_matrix = np.array(s_list)
    np.save('s_matrix', s_matrix)

    plt.figure()
    plt.matshow(s_matrix)
    plt.title('Singular Values')
    plt.ylabel('N starts at 200\n, ends at 10,000 \nin steps of 1,000 \nincreasing from top to bottom')
    plt.xlabel('Singular value')
    plt.show()


def compressive_proj(m, X, PHI):
    p = 1/math.sqrt(m) * np.matmul(PHI, X)
    return p

def generate_PHI(m, d):
    return np.random.choice([+1, -1], p=[1 / 2, 1 / 2], size=[m, d])

def define_basis(m, d):
    return sample_generation.generate_d_dim_samples(d, m, best_seeds=np.load('best_seeds.npy'))

if __name__ == '__main__':
    '''Work in progress'''


    # Generate quasi-orthogonal basis
    q_orthoganal_basis = ul.find_quasi_orthogonal_basis(dim=100, num_basis=6, from_memory=True)

    # plot_s_wrt_N(q_orthoganal_basis)


    d_dim_samples, one_hot_labels = ul.generate_d_dim_data(q_orthoganal_basis, num_samples=250)

    pca = PCA(n_components=6) # pick first 6
    pca.fit(d_dim_samples)

    U, S, VT = np.linalg.svd(d_dim_samples - d_dim_samples.mean(0))

    X_train_pca = pca.transform(d_dim_samples)

    # Run K-Means algorithm with K_means++ initialization
    d_dim_kmeans_tests = ul.run_Kmeans_algorithm(X_train_pca, one_hot_labels, rangeK=np.arange(2, 6),
                                              k_means_plusplus=True, plot_tables=False)
    ul.get_geometric_insight(d_dim_kmeans_tests, q_orthoganal_basis)

    loss_lst = []
    for m in range(1, 50):
        # Reconstruction with L1 (Lasso) penalization
        # the best value of alpha was determined using cross validation
        # with LassoCV
        i  = i/100
        X_projected = pca.inverse_transform(X_train_pca)
        loss = ((d_dim_samples - X_projected) ** 2).mean()
        loss_lst.append(loss)

    d_dim_samples, one_hot_labels, noise = ul.generate_d_dim_data(q_orthoganal_basis, num_samples=250, seperate_noise=True)
    d = 100
    for m in range (10, 200, 10):
        PHI = generate_PHI(m, d)
        B = define_basis(m, d)
        p = compressive_proj(m, d_dim_samples + noise, PHI)

    s_hat = solve_lasso(P, m, PHI, B, )

    # Reconstruction with L1 (Lasso) penalization
    # the best value of alpha was determined using cross validation
    # with LassoCV


    #
    # # Run K-Means algorithm with K_means++ initialization
    # d_dim_kmeans_tests = ul.run_Kmeans_algorithm(d_dim_samples, one_hot_labels, rangeK=np.arange(2, 6),
    #                                           k_means_plusplus=True, plot_tables=False)
    # ul.get_geometric_insight(d_dim_kmeans_tests, q_orthoganal_basis)
    #
    # # Run EM algorithm to estimate gaussian mixture parameters
    # d_dim_EM_tests = ul.run_EM_algorithm(d_dim_kmeans_tests, one_hot_labels, plot_tables=False, epsilon=0.001)
    # ul.get_geometric_insight(d_dim_EM_tests, q_orthoganal_basis)