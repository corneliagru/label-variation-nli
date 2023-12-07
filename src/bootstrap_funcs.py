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

# N, C = Y.shape
# pi = np.zeros((B, K))
# theta = np.zeros((B, K, C))
# tau = np.zeros((B, N, K))
#
# for b in range(B):
#     # Sample with replacement
#     Y_b = np.zeros((N, C))
#     for n in range(N):
#         Y_b[n] = Y[random.randint(0, N - 1)]
#
#     # Run EM
#     em = EM(K=K)
#     _, pi_b, theta_b, tau_b = em.fit(Y_b)
#
#     # Save results
#     pi[b] = pi_b
#     theta[b] = theta_b
#     tau[b] = tau_b
#
# return pi, theta, tau
