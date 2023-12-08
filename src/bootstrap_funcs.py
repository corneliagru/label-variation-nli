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

    B_sample = []
    B_pi = []
    B_theta = []
    B_tau = []
    B_bounds = []

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

        B_sample.append(sample)
        B_pi.append(pi_matched)
        B_theta.append(theta_matched)
        B_tau.append(tau_matched)
        B_bounds.append(bounds)
    print('Bootstrap finished.')

    return {'pi': B_pi, 'theta': B_theta, 'tau': B_tau, 'sample': B_sample, 'bounds': B_bounds}

def freq_to_matrix(Y):
    mat = np.empty((len(Y),100))
    i=0
    for obs in Y:
        arr = np.empty(100, dtype=int)
        current_index = 0
        for category, count in enumerate(obs):
            arr[current_index:count+current_index] = category
            current_index += count
        mat[i] = arr
        i+=1
    return mat

def N_J_bootstrap_em(N_J_comb, snli_annotations, B=50, seed=12):
    np.random.seed(seed)

    N, J = N_J_comb

    B_samples_N_J, B_pi_N_J, B_theta_N_J, B_tau_N_J, B_bounds_N_J = [], [], [], [], []

    for b in range(B):
        if b%10 == 0:
            print('Bootstrap iteration: ', b)

        if J < 100:
            random_J = np.random.choice(100, size=J, replace=False)
            annotations_J = snli_annotations[:,random_J].astype(int)
            count_matrix = np.zeros((3, len(annotations_J)), dtype=int)

            for row_idx in range(annotations_J.shape[0]):
                row = annotations_J[row_idx]
                unique, counts = np.unique(row, return_counts=True)
                count_matrix[unique, row_idx] = counts

            snli_one_hot_J = count_matrix.T

        else:
            snli_one_hot_J = snli_one_hot_arr

        b_sample = np.array(random.choices(snli_one_hot_J, k=N))

        try:
            b_snli_em = run_em(b_sample, K=3)
        except:
            print('Error in EM')
            continue

        matched_index = np.argsort(np.argmax(b_snli_em[2], axis=1))
        b_pi_matched = b_snli_em[1][matched_index]
        b_theta_matched = b_snli_em[2][matched_index]
        b_tau_matched = b_snli_em[3][:, matched_index]
        b_bounds = get_boundaries(b_pi_matched, b_theta_matched)

        B_samples_N_J.append(b_sample)
        B_pi_N_J.append(b_pi_matched)
        B_theta_N_J.append(b_theta_matched)
        B_tau_N_J.append(b_tau_matched)
        B_bounds_N_J.append(b_bounds)

    return B_pi_N_J, B_theta_N_J, B_tau_N_J, B_samples_N_J, B_bounds_N_J, N_J_comb