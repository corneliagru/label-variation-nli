from src.utils import common, list_dict_data_tool
from scipy.stats import multinomial, dirichlet
import numpy as np
import pandas as pd


# python-ternary needed


def load_data_as_df(data_file):
    """
    Loads data from jsonl file into pandas dataframe
    :param data_file: file path to jsonl file
    :return: pandas dataframe
    """
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
        """
        Computes log likelihood of the data given the model
        :param Y: data
        :param pi: pi parameter of multinomial
        :param theta: theta parameter of multinomial
        :return: log likelihood
        """
        mn_probs = np.zeros(Y.shape[0])
        for k in range(theta.shape[0]):
            mn_probs_k = pi[k] * self._multinomial_prob(Y, theta[k])
            mn_probs += mn_probs_k
        mn_probs[mn_probs == 0] = np.finfo(float).eps
        return np.log(mn_probs).sum()

    def _multinomial_prob(self, counts, theta):
        """
        Computes multinomial probability
        :param counts: vector of counts of dimension (C)
        :param theta: theta vector of multinomial parameters of dimension (C)
        :return: probability of the observation given the respective theta
        """

        n = counts.sum(axis=-1)
        m = multinomial(n, theta)
        return m.pmf(counts)

    def _e_step(self, Y, pi, theta):
        """
        Performs E-step, i.e., computes posterior probability as ((prior * likelihood)/evidence)
        :param Y: data points of dimension (N x C)
        :param pi: pi parameter, i.e., mixture weights of dimension (K)
        :param theta: theta parameter, i.e., multinomial probabilities of dimension (K x C)
        :return: tau: probabilities of classes for data points of dimension (N x K)
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
        Performs M-step, recomputes parameters given posterior classes,
        i.e., new pi is relative frequency of posterior class (tau) of all votes (Y)
        :param Y: data points of dimension (N x C)
        :param tau: probabilities of classes for data points of dimension (N x K)
        :return: pi, theta. pi: mixture weights of dimension (K), theta: multinomial probabilities of dimension (K x C)
        """

        # Compute pi
        pi = tau.sum(axis=0) / tau.sum()

        # Compute theta
        weighted_counts = tau.T.dot(Y)
        theta = weighted_counts / weighted_counts.sum(axis=-1).reshape(-1, 1)

        return pi, theta

    def _compute_loss(self, Y, pi, theta, tau):
        """
        Computes loss of the model given the data and parameters. Each input is a numpy array.
        :param Y: data points of dimension (N x C)
        :param pi: pi parameter, i.e., mixture weights of dimension (K)
        :param theta: theta parameter, i.e., multinomial probabilities of dimension (K x C)
        :param tau: tau parameter, i.e., probabilities of classes for data points of dimension (N x K)
        :return:
        """

        loss = 0
        for k in range(pi.shape[0]):
            weights = tau[:, k]
            loss += np.sum(weights * (np.log(pi[k]) + np.log(self._multinomial_prob(Y, theta[k]))))
            loss -= np.sum(weights * np.log(weights))
        return loss

    # initialize pi as uniform distribution, theta (= confusion values) as dirichlet since its the conjugate prior to
    # a multinomial
    def _init_params(self, C):
        """
        Initializes parameters for EM algorithm. pi is initialized as equal probabilities, i.e., 1/K ,
        theta (= confusion values) is sampled K (number of latent classes) times from a dirichlet since it's the
        conjugate prior to a multinomial.
        Parameter alpha has C entries, where each value is 2·C. C is the number of observed classes in the data.
        :param C: number of observed classes in data
        :return: pi, theta
        """
        pi = np.array([1 / self._K] * self._K)
        theta = dirichlet.rvs([2 * C] * C, self._K)
        return pi, theta

    def _train_once(self, Y):
        """
        Trains the model once. Initializes parameters, then performs E-step and M-step until convergence.
        :param Y: data points of dimension (N x C)
        :return: pi, theta, tau, loss
        """
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
        """
        Fits the model to the data. Performs training multiple times (restarts) and returns the best model.
        :param Y: data points of dimension (N x C)
        :return: best_loss, best_pi, best_theta, best_tau
        """
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
    """
    Runs EM algorithm on data Y with K latent classes. Returns the best model.
    :param Y: data points of dimension (N x C)
    :param K: number of latent classes
    :return: likelihoods, best_pi, best_theta, best_tau
    """
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

    return likelihoods, best_pi, best_theta, best_tau


def tolerance_sort(array, tolerance, reverse=False):
    """
    Sorts an array by the first column, but only if the difference between the rows is more than the tolerance.
    Otherwise, tolerance_sort by the next column.
    Usually, tolerance sorting is only needed if more latent classes should be estimated than observed classes.
    If K=C, regular sorting is sufficient, and is implicitly done in tolerance_sort.
    :param array: array to sort
    :param tolerance: tolerance for regular sorting
    :param reverse: sort descending if TRUE, ascending otheriwse
    :return: sorted array, matched index
    """
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
    """
    Returns the label with the highest posterior probability
    :param tau: input array of posterior probabilities
    :return: discrete label of highest posterior probability
    """
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
