from src.utils import common, list_dict_data_tool
from scipy.stats import multinomial, dirichlet
import numpy as np
import pandas as pd
import ternary
from sympy import symbols, solve


def load_data_as_df(data_file):
    d_list = common.load_jsonl(data_file)
    collected_data_dict = list_dict_data_tool.list_to_dict(d_list, key_fields='uid')
    df = pd.DataFrame.from_dict(collected_data_dict, orient='index')
    return df


class MultinomialExpectationMaximizer:
    def __init__(self, K, rtol=1e-4, max_iter=100, restarts=10):
        self._K = K
        self._rtol = rtol
        self._max_iter = max_iter
        self._restarts = restarts

    def log_lh(self, Y, pi, theta):
        mn_probs = np.zeros(Y.shape[0])
        for k in range(theta.shape[0]):
            mn_probs_k = pi[k] * self._multinomial_prob(Y, theta[k])
            mn_probs += mn_probs_k
        mn_probs[mn_probs == 0] = np.finfo(float).eps
        return np.log(mn_probs).sum()

    def _multinomial_prob(self, counts, theta):
        """
        counts: (C), vector of counts
        theta: (C), vector of multinomial parameters

        Returns:
        p: (1), probability of the observation given the respective theta
        """
        n = counts.sum(axis=-1)
        m = multinomial(n, theta)
        return m.pmf(counts)

    def _e_step(self, Y, pi, theta):
        """
        Performs E-step, i.e., computes posterior probability as ((prior * likelihood)/evidence)
        Y: (N x C), data points
        pi: (K), mixture weights
        theta: (K x C), multinomial probabilities

        Returns:
        tau: (N x K), probabilities of classes for objects
        """
        # Compute tau
        N = Y.shape[0]
        K = pi.shape[0]
        weighted_multi_prob = np.zeros((N, K))
        for k in range(K):
            weighted_multi_prob[:, k] = pi[k] * self._multinomial_prob(Y, theta[k])

        denum = weighted_multi_prob.sum(axis=1)
        tau = weighted_multi_prob / denum.reshape(-1, 1)

        return tau

    def _m_step(self, Y, tau):
        """
        Performs M-step, i.e., pi is relative posterior (tau) of all votes (Y)
        Y: (N x C), data points
        tau: (N x K), probabilities of classes for objects

        Returns:
        pi: (K), mixture weights
        theta: (K x C), multinomial probabilities
        """
        # Compute pi
        pi = tau.sum(axis=0) / tau.sum()

        # Compute theta
        weighted_counts = tau.T.dot(Y)
        theta = weighted_counts / weighted_counts.sum(axis=-1).reshape(-1, 1)

        return pi, theta

    def _compute_loss(self, Y, pi, theta, tau):
        """
        Each input is numpy array:
        Y: (N x C), data points
        pi: (K), mixture component weights
        theta: (K x C), multinomial categories weights
        tau: (N x K), probabilities of clusters for objects
        """
        loss = 0
        for k in range(pi.shape[0]):
            weights = tau[:, k]
            loss += np.sum(weights * (np.log(pi[k]) + np.log(self._multinomial_prob(Y, theta[k]))))
            loss -= np.sum(weights * np.log(weights))
        return loss

    # initialize pi as uniform distribution, thetha (= confusion values) as dirichlet since its the cojugate prior to a multinomial
    def _init_params(self, C):
        pi = np.array([1 / self._K] * self._K)
        theta = dirichlet.rvs([2 * C] * C, self._K)
        return pi, theta

    def _train_once(self, Y):
        loss = float('inf')
        C = Y.shape[1]
        pi, theta = self._init_params(C)

        for it in range(self._max_iter):
            prev_loss = loss
            tau = self._e_step(Y, pi, theta)
            pi, theta = self._m_step(Y, tau)
            loss = self._compute_loss(Y, pi, theta, tau)
            if it > 0 and (np.abs((prev_loss - loss) / prev_loss) < self._rtol):
                break
        return pi, theta, tau, loss

    def fit(self, Y):
        best_loss = -float('inf')
        best_pi = None
        best_theta = None
        best_tau = None

        for it in range(self._restarts):
            pi, theta, tau, loss = self._train_once(Y)
            if loss > best_loss:
                best_loss = loss
                best_pi = pi
                best_theta = theta
                best_tau = tau

        return best_loss, best_pi, best_theta, best_tau


def run_em(Y, K=3):
    likelihoods = []
    best_pi = None
    best_theta = None
    best_tau = None

    model = MultinomialExpectationMaximizer(K, restarts=100)
    _, pi, theta, tau = model.fit(Y)
    log_likelihood = model.log_lh(Y, pi, theta)
    likelihoods.append(log_likelihood)
    best_pi = pi
    best_theta = theta
    best_tau = tau

    # print('best_pi: %s' % str(best_pi))
    # print('best_theta: %s' % str(best_theta))

    return likelihoods, best_pi, best_theta, best_tau


# Label matching
# based on rel. frequency and estimated pi should be sufficient
def sort_second_array(first_array, second_array):
    r = np.argsort(first_array)
    p = np.argsort(second_array)

    matched_index = [x for _, x in sorted(zip(r, p))]

    return matched_index


def tolerance_sort(array, tolerance, reverse=False):
    '''
    Sorts an array by the first column, but only if the difference between the rows is more than the tolerance.
    Otherwise, tolerance_sort by the next column.
    Usually, tolerance sorting is only needed if more latent classes should be estimated than observed classes.
    If K=C, regular sorting is sufficient, and is implicitly done in tolerance_sort.
    :param array: array to sort
    :param tolerance: tolerance for regular sorting
    :param reverse: sort descending if TRUE, ascending otheriwse
    :return: sorted array, matched index
    '''
    array_sorted = np.copy(array[np.lexsort((array[:, 1], array[:, 0]))])
    sort_range = [0]
    for i in range(array.shape[0] - 1):
        if array_sorted[i + 1, 0] - array_sorted[i, 0] <= tolerance:
            sort_range.append(i + 1)
            continue
        else:
            sub_arr = np.take(array_sorted, sort_range, axis=0)
            sub_arr_ord = np.copy(
                sub_arr[np.lexsort((sub_arr[:, 0], sub_arr[:, 1]))])
            array_sorted[slice(sort_range[0], sort_range[-1] +
                               1)] = sub_arr_ord
            sort_range = [i + 1]
    if (reverse): array_sorted = np.flipud(array_sorted)

    matched_index = [np.where(np.all(array == x, axis=1))[0][0] for x in array_sorted]
    return array_sorted, matched_index


def max_posterior_label(tau):
    '''
    Returns the label with the highest posterior probability
    :param tau: input array of posterior probabilities
    :return: discrete label of highest posterior probability
    '''
    ncol = np.shape(tau)[1]
    z_int = np.argmax(tau, axis=1)

    if ncol == 3:
        # mapping is {0: 'e', 1: 'n', 2: 'c'}
        z_hat = np.where(z_int == 0, 'e',
                         np.where(z_int == 1, 'n',
                                  np.where(z_int == 2, 'c', z_int)))
    elif ncol == 4:
        z_hat = np.where(z_int == 0, 'e',
                         np.where(z_int == 1, 'en',
                                  np.where(z_int == 2, 'nc',
                                           np.where(z_int == 3, 'c', z_int))))
    elif ncol == 5:
        z_hat = np.where(z_int == 0, 'e',
                         np.where(z_int == 1, 'en',
                                  np.where(z_int == 2, 'n',
                                           np.where(z_int == 3, 'nc',
                                                    np.where(z_int == 4, 'c', z_int)))))
    return z_hat


CB_color_cycle = {0: '#377eb8', 1: '#ff7f00', 2: '#4daf4a',
                  3: '#f781bf', 4: '#a65628', 5: '#984ea3',
                  6: '#999999', 7: '#e41a1c', 8: '#dede00'}


def base_plot(scale=100, fontsize=12, size=6, multiple=10, tick_fontsize=12, tick_offset=0.02, label_offset=0):
    '''
    Generates a base ternary plot with the given scale and fontsize
    :param scale: maximum value of the plot
    :param fontsize: fontsize of the labels
    :return: figure and tax object
    '''
    # function to generate base ternary plot
    figure, tax = ternary.figure(scale=scale)
    figure.set_size_inches(size, size)
    midpoint = (33.33, 33.33, 33.33)
    tax.line((50, 50, 0), midpoint, color="black")
    tax.line((0, 50, 50), midpoint, color="black")
    tax.line((50, 0, 50), midpoint, color="black")
    # counterclockwise drawing
    tax.right_corner_label("entailment", fontsize=fontsize, offset=label_offset)
    tax.top_corner_label("neutral", fontsize=fontsize)
    tax.left_corner_label("contradiction", fontsize=fontsize, offset=label_offset)
    tax.boundary()
    tax.gridlines(multiple=multiple)
    tax.get_axes().axis('off')
    tax.ticks(axis='lbr', linewidth=1, multiple=multiple, fontsize=tick_fontsize, offset=tick_offset)
    tax.clear_matplotlib_ticks()
    return figure, tax


def create_all_labels(n_labels):
    all_labels = []
    for i in range(n_labels + 1):
        for j in range(n_labels + 1):
            for k in range(n_labels + 1):
                if i + j + k == n_labels:
                    all_labels.append([i, j, k])
    return np.array(all_labels)


def get_boundaries(pi, theta):
    y_e, y_c, y_n = symbols('y_e y_c y_n')

    # e-c
    eq1 = y_e + y_c - 100
    eq3 = y_n
    eq2 = (np.log(pi[0]) + y_e * np.log(theta[0, 0]) + y_n * np.log(theta[0, 1]) + y_c * np.log(theta[0, 2])) - (
                np.log(pi[2]) + y_e * np.log(theta[2, 0]) + y_n * np.log(theta[2, 1]) + y_c * np.log(theta[2, 2]))
    dict_ec = solve((eq1, eq2, eq3), (y_e, y_c, y_n))

    # e-n
    eq1 = y_e + y_n - 100
    eq3 = y_c
    eq2 = (np.log(pi[0]) + y_e * np.log(theta[0, 0]) + y_n * np.log(theta[0, 1]) + y_c * np.log(theta[0, 2])) - (
                np.log(pi[1]) + y_e * np.log(theta[1, 0]) + y_n * np.log(theta[1, 1]) + y_c * np.log(theta[1, 2]))
    dict_en = solve((eq1, eq2, eq3), (y_e, y_c, y_n))

    # n-c
    eq1 = y_n + y_c - 100
    eq3 = y_e
    eq2 = (np.log(pi[2]) + y_e * np.log(theta[2, 0]) + y_n * np.log(theta[2, 1]) + y_c * np.log(theta[2, 2])) - (
                np.log(pi[1]) + y_e * np.log(theta[1, 0]) + y_n * np.log(theta[1, 1]) + y_c * np.log(theta[1, 2]))
    dict_nc = solve((eq1, eq2, eq3), (y_e, y_c, y_n))

    # e-n-c
    eq1 = y_e + y_c + y_n - 100
    eq3 = (np.log(pi[0]) + y_e * np.log(theta[0, 0]) + y_n * np.log(theta[0, 1]) + y_c * np.log(theta[0, 2])) - (
                np.log(pi[1]) + y_e * np.log(theta[1, 0]) + y_n * np.log(theta[1, 1]) + y_c * np.log(theta[1, 2]))
    eq2 = (np.log(pi[0]) + y_e * np.log(theta[0, 0]) + y_n * np.log(theta[0, 1]) + y_c * np.log(theta[0, 2])) - (
                np.log(pi[2]) + y_e * np.log(theta[2, 0]) + y_n * np.log(theta[2, 1]) + y_c * np.log(theta[2, 2]))

    dict_enc = solve((eq1, eq2, eq3), (y_e, y_c, y_n))

    # turn to coordinates
    point_ec = [dict_ec[y_e], dict_ec[y_n], dict_ec[y_c]]
    point_nc = [dict_nc[y_e], dict_nc[y_n], dict_nc[y_c]]
    point_en = [dict_en[y_e], dict_en[y_n], dict_en[y_c]]
    point_enc = [dict_enc[y_e], dict_enc[y_n], dict_enc[y_c]]

    return point_ec, point_en, point_nc, point_enc


def col_to_label(tau):
    if pd.Series(np.argmax(tau, axis=1)).map(CB_color_cycle) == '#377eb8':
        return 'entailment'
    elif pd.Series(np.argmax(tau, axis=1)).map(CB_color_cycle) == '#ff7f00':
        return 'neutral'
    elif pd.Series(np.argmax(tau, axis=1)).map(CB_color_cycle) == '#4daf4a':
        return 'contradiction'


def ternary_plot(Y, tau, pi, theta, scatter_visible=True, all=True, bounds=True, bounds_col=CB_color_cycle[7],
                 alpha=0.5, scale=100, fontsize=12, size=6, tick_fontsize=12, tick_offset=0.02, label_offset=0):
    figure, tax = base_plot(scale=scale, fontsize=fontsize, size=size, tick_fontsize=tick_fontsize,
                            tick_offset=tick_offset, label_offset=label_offset)

    if scatter_visible:
        if all:
            Y_all = create_all_labels(100)
            tau_all = MultinomialExpectationMaximizer(K=3)._e_step(Y_all, pi=pi, theta=theta)
            tax.scatter(Y_all, c=pd.Series(np.argmax(tau_all, axis=1)).map(CB_color_cycle))
        else:
            tax.scatter(Y, c=pd.Series(np.argmax(tau, axis=1)).map(CB_color_cycle), alpha=alpha)

    if bounds:
        bounds = get_boundaries(pi=pi, theta=theta)
        tax.line(bounds[0], bounds[3], color=bounds_col)
        tax.line(bounds[1], bounds[3], color=bounds_col)
        tax.line(bounds[2], bounds[3], color=bounds_col)

    return figure, tax


def multiple_boundaries(pi_list, theta_list, bounds_original):
    figure, tax = base_plot()

    for i in range(len(pi_list)):
        bounds = get_boundaries(pi=pi_list[i], theta=theta_list[i])

        tax.line(bounds[0], bounds[3], color="gray")
        tax.line(bounds[1], bounds[3], color="gray")
        tax.line(bounds[2], bounds[3], color="gray")

    tax.line(bounds_original[0], bounds_original[3], color="red")
    tax.line(bounds_original[1], bounds_original[3], color="red")
    tax.line(bounds_original[2], bounds_original[3], color="red")

    return figure, tax


def multiple_boundaries_colored(pi_list, theta_list, bounds_original, cmap):
    figure, tax = base_plot()
    cols = mcp.gen_color(cmap=cmap, n=len(pi_list) + 10)

    for i in range(len(pi_list)):
        bounds = get_boundaries(pi=pi_list[i], theta=theta_list[i])

        tax.line(bounds[0], bounds[3], color=cols[i + 6])
        tax.line(bounds[1], bounds[3], color=cols[i + 6])
        tax.line(bounds[2], bounds[3], color=cols[i + 6])

    tax.line(bounds_original[0], bounds_original[3], color="red")
    tax.line(bounds_original[1], bounds_original[3], color="red")
    tax.line(bounds_original[2], bounds_original[3], color="red")

    return figure, tax
