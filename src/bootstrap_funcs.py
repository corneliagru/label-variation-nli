# Imports

import numpy as np
import random

from src.model_funcs import run_em, tolerance_sort
from src.plotting_funcs import get_boundaries


def bootstrap(Y, N=None, K=3, B=50, seed=12):
    """
    Each input is numpy array:
    Y: (N x C), data points
    K: number of clusters
    B: number of bootstrap samples

    Returns:
    pi: (K), mixture component weights
    theta: (K x C), multinomial categories weights
    tau: (N x K), probabilities of clusters for objects
    """

    b_sample = []
    b_pi = []
    b_theta = []
    b_tau = []
    b_bounds = []

    if N is None:
        N = Y.shape[0]

    random.seed(seed)
    for b in range(B):

        if b % 10 == 0:
            print('Bootstrap iteration: ', b, '/', B)

        sample = np.array(random.choices(Y, k=N))
        em_result = run_em(sample, K=K)

        theta_matched, matched_index = tolerance_sort(em_result[2], 0.05, reverse=True)

        pi_matched = em_result[1][matched_index]
        tau_matched = em_result[3][:, matched_index]
        bounds = get_boundaries(pi_matched, theta_matched)

        b_sample.append(sample)
        b_pi.append(pi_matched)
        b_theta.append(theta_matched)
        b_tau.append(tau_matched)
        b_bounds.append(bounds)
    print('Bootstrap finished.')

    return {'pi': b_pi, 'theta': b_theta, 'tau': b_tau, 'sample': b_sample, 'bounds': b_bounds}


def freq_to_matrix(Y):
    """
     function to convert frequency matrix to matrix of individual labels
    :param Y: frequency matrix
    :return: matrix of individual labels
    """
    mat = np.empty((len(Y), 100))
    i = 0
    for obs in Y:
        arr = np.empty(100, dtype=int)
        current_index = 0
        for category, count in enumerate(obs):
            arr[current_index:count + current_index] = category
            current_index += count
        mat[i] = arr
        i += 1
    return mat


def N_J_bootstrap_em(Y, N_J_comb, snli_annotations, B=50, seed=12):
    """
    function to run bootstrap on N and J combinations
    :param Y: frequency matrix
    :param N_J_comb: (N,J) combinations to run bootstrap on
    :param snli_annotations: matrix of individual labels
    :param B: number of bootstrap iterations
    :param seed: random seed
    :return: list of results for each bootstrap iteration. pi, theta, tau, b_samples, b_bounds, N_J_comb
    """
    np.random.seed(seed)

    N, J = N_J_comb

    b_samples, b_pi, b_theta, b_tau, b_bounds = [], [], [], [], []

    for b in range(B):
        if b % 10 == 0:
            print('Bootstrap iteration: ', b)

        if J < 100:
            index_J = np.random.choice(100, size=J, replace=False)
            annotations_J = snli_annotations[:, index_J].astype(int)
            count_matrix = np.zeros((3, len(annotations_J)), dtype=int)

            for row_idx in range(annotations_J.shape[0]):
                row = annotations_J[row_idx]
                unique, counts = np.unique(row, return_counts=True)
                count_matrix[unique, row_idx] = counts

            snli_one_hot_J = count_matrix.T

        else:
            snli_one_hot_J = Y

        b_sample = np.array(random.choices(snli_one_hot_J, k=N))

        try:
            b_em = run_em(b_sample, K=3)
        except:
            print('Error in EM')
            continue

        matched_index = np.argsort(np.argmax(b_em[2], axis=1))
        b_pi_matched = b_em[1][matched_index]
        b_theta_matched = b_em[2][matched_index]
        b_tau_matched = b_em[3][:, matched_index]
        b_bound = get_boundaries(b_pi_matched, b_theta_matched)

        b_samples.append(b_sample)
        b_pi.append(b_pi_matched)
        b_theta.append(b_theta_matched)
        b_tau.append(b_tau_matched)
        b_bounds.append(b_bound)

    return b_pi, b_theta, b_tau, b_samples, b_bounds, N_J_comb
