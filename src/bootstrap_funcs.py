# Imports

import numpy as np
import random

from src.model_funcs import run_em, tolerance_sort
from src.plotting_funcs import get_boundaries


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


def N_J_bootstrap_em(Y, N_J_comb, snli_annotations, B=50):
    """
    function to run bootstrap on N and J combinations
    :param Y: frequency matrix, i.e., data points of dimension (N x C)
    :param N_J_comb: (N,J) combinations to run bootstrap on
    :param snli_annotations: matrix of individual labels
    :param B: number of bootstrap iterations
    :return: list of results for each bootstrap iteration. pi, theta, tau, b_samples, b_bounds, N_J_comb
    """

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
