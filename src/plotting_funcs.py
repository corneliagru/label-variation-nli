import numpy as np
import pandas as pd
import ternary
from sympy import symbols, solve

from src.model_funcs import MultinomialExpectationMaximizer

CB_color_cycle = {0: '#377eb8', 1: '#ff7f00', 2: '#4daf4a',
                  3: '#f781bf', 4: '#a65628', 5: '#984ea3',
                  6: '#999999', 7: '#e41a1c', 8: '#dede00'}


def plot_ternary_axes(scale=100, fontsize=12, size=6, multiple=10, multiple_grid=10,  tick_fontsize=12,
                      tick_offset=0.02, label_offset=0, labels=["entailment", "neutral", "contradiction"],
                      weight='normal'):
    """
    Generates a base ternary plot
    :param multiple_grid: which multiple of the grid should be plotted
    :param scale: maximum possible value of the plot
    :param fontsize: fontsize of the labels
    :param size: size of plot in inches
    :param multiple: which multiples of the scale to plot
    :param tick_fontsize: fontsize of the ticks
    :param tick_offset: offset of the ticks
    :param label_offset: offset of the labels
    :param labels: label names of the axes in order of right, top, left
    :param weight: weight of the labels font (i.e., bold or normal, etc.)
    :return: figure, tax object
    """

    # function to generate base ternary plot
    figure, tax = ternary.figure(scale=scale)
    figure.set_size_inches(size, size)
    midpoint = (33.33, 33.33, 33.33)
    tax.line((50, 50, 0), midpoint, color="black")
    tax.line((0, 50, 50), midpoint, color="black")
    tax.line((50, 0, 50), midpoint, color="black")
    # counterclockwise drawing
    tax.right_corner_label(labels[0], fontsize=fontsize, offset=label_offset, weight=weight)
    tax.top_corner_label(labels[1], fontsize=fontsize, weight=weight)
    tax.left_corner_label(labels[2], fontsize=fontsize, offset=label_offset, weight=weight)
    tax.boundary()
    tax.gridlines(multiple=multiple_grid)
    tax.get_axes().axis('off')
    tax.ticks(axis='lbr', linewidth=1, multiple=multiple, fontsize=tick_fontsize, offset=tick_offset)
    tax.clear_matplotlib_ticks()
    return figure, tax


def create_all_labels(n_labels):
    """
    Creates all possible label combinations for a given number of labels for three classes.
    :param n_labels: maximum value of votes
    :return: vector of all label combinations
    """
    all_labels = []
    for i in range(n_labels + 1):
        for j in range(n_labels + 1):
            for k in range(n_labels + 1):
                if i + j + k == n_labels:
                    all_labels.append([i, j, k])
    return np.array(all_labels)


def get_boundaries(pi, theta):
    """
    Computes boundaries of the simplex for a given pi and theta
    :param pi: pi parameter of multinomial mixture model
    :param theta: theta parameter of multinomial mixture model
    :return: end points of the boundaries
    """
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
    """
    Converts color to label
    :param tau: posterior probabilities
    :return: string of label name
    """
    if pd.Series(np.argmax(tau, axis=1)).map(CB_color_cycle) == '#377eb8':
        return 'entailment'
    elif pd.Series(np.argmax(tau, axis=1)).map(CB_color_cycle) == '#ff7f00':
        return 'neutral'
    elif pd.Series(np.argmax(tau, axis=1)).map(CB_color_cycle) == '#4daf4a':
        return 'contradiction'


def plot_ternary_scatter(tax, Y, tau, pi, theta, plot_all=True, K=3, alpha=0.5):
    """
    Plots scatter into a ternary plot
    :param tax: axes object of ternary plot as generated by plot_ternary_axes
    :param Y: data points
    :param tau: tau values
    :param pi: pi values
    :param theta: theta values
    :param plot_all: plot all possible label combinations if TRUE, otherwise plot only the labels in Y
    :param K: number of latent classes
    :param alpha: intensity of the scatter
    :return: tax object
    """
    if plot_all:
        Y_all = create_all_labels(Y.shape[1])
        tau_all = MultinomialExpectationMaximizer(K=K)._e_step(Y_all, pi=pi, theta=theta)
        tax.scatter(Y_all, c=pd.Series(np.argmax(tau_all, axis=1)).map(CB_color_cycle))
    else:
        tax.scatter(Y, c=pd.Series(np.argmax(tau, axis=1)).map(CB_color_cycle), alpha=alpha)
    return tax


def plot_ternary_bounds(tax, pi, theta, bounds_col=CB_color_cycle[7]):
    """
    plots the estimated class boundaries for a given pi and theta
    :param tax: axes object of ternary plot as generated by plot_ternary_axes
    :param pi: pi parameter of multinomial mixture model
    :param theta: theta parameter of multinomial mixture model
    :param bounds_col: color of the boundaries
    :return: tax object
    """
    bounds = get_boundaries(pi=pi, theta=theta)
    tax.line(bounds[0], bounds[3], color=bounds_col)
    tax.line(bounds[1], bounds[3], color=bounds_col)
    tax.line(bounds[2], bounds[3], color=bounds_col)
    return tax


def plot_bootstrap_bounds(tax, bounds_list, alpha=0.7):
    """
    Plots all booostrapped bounds in gray and the range of the bounds in orange at axes
    :param tax: axes object of ternary plot as generated by plot_ternary_axes
    :param bounds_list: list of bounds
    :param alpha: intensity of the gray lines
    :return: tax object
    """
    # contradiction vs entailment
    coordinates_ce = [x[0] for x in bounds_list]
    max_index_ce = np.argmax(coordinates_ce, axis=0)
    pt_ce_1 = coordinates_ce[max_index_ce[0]]
    pt_ce_2 = coordinates_ce[max_index_ce[2]]

    # contradiction vs neutral
    coordinates_cn = [x[1] for x in bounds_list]
    max_index_cn = np.argmax(coordinates_cn, axis=0)
    pt_cn_1 = coordinates_cn[max_index_cn[0]]
    pt_cn_2 = coordinates_cn[max_index_cn[1]]

    # neutral vs entailment
    coordinates_ne = [x[2] for x in bounds_list]
    max_index_ne = np.argmax(coordinates_ne, axis=0)
    pt_ne_1 = coordinates_ne[max_index_ne[1]]
    pt_ne_2 = coordinates_ne[max_index_ne[2]]

    for b in range(len(bounds_list)):
        bnd = bounds_list[b]
        if len(bnd) == 4:
            tax.line(bnd[0], bnd[3], color="gray", linewidth=0.7, alpha=alpha)
            tax.line(bnd[1], bnd[3], color="gray", linewidth=0.7, alpha=alpha)
            tax.line(bnd[2], bnd[3], color="gray", linewidth=0.7, alpha=alpha)

        else:
            tax.line(bnd[2], bnd[3], color="gray", linewidth=0.7, alpha=alpha)
            tax.line(bnd[1], bnd[4], color="gray", linewidth=0.7, alpha=alpha)

    # plot the range of the bounds in orange at axes
    tax.line(pt_ce_1, pt_ce_2, color=CB_color_cycle[1], linewidth=3)
    tax.line(pt_cn_1, pt_cn_2, color=CB_color_cycle[1], linewidth=3)
    tax.line(pt_ne_1, pt_ne_2, color=CB_color_cycle[1], linewidth=3)

    return tax
