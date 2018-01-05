#!/usr/bin/env python3
"""Performance evaluation of ternary classification.

Functions:
    get_sas: Get the sensitivity, accuracy and specificity for a
        ternary classification result as 2-d arrays.
    get_sas_list: Get SAS lists from a weighted network.
    plot_sas: Plot or save AS and SS curves and metrics.
    get_metric: Calculate classification metric.
    hmean: Calculate the harmonic mean.
    integrate_lin: Integration with linear interpolation.
"""

import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

def get_sas(decision, prior, self_edge=False):
    """Get sensitivity, accuracy and specificity for a ternary
    classification decision.

    Args:
        decision: A 2-d array with 1, -1 and 0s representing the
            classification decision.
        prior: A 2-d array with 1, -1 and 0s representing the
            ground truth.
        self_edge: Indicator of whether self-edges are allowed.
            Default is false; i.e., self-edges in the ground thuth
            are ignored and not counted toward the sensitivity,
            accuracy or specificity.

    Returns:
        A 3-tuple of the sensitivity, accuracy and specificity,
            with NaN if division by zero occurs.
    """
    prior_flat = np.copy(prior)
    decision_flat = np.copy(decision)
    if not self_edge:
        np.fill_diagonal(prior_flat, np.nan)
        np.fill_diagonal(decision_flat, np.nan)
    # Convert 2-d arrays to 1-d arrays.
    prior_flat = prior_flat[np.isfinite(prior_flat)]
    decision_flat = decision_flat[np.isfinite(decision_flat)]
    # True positive.
    tp = sum(prior_flat*decision_flat > 0)
    # Number of detected edges.
    num_detect = sum(decision_flat != 0)
    # Positive.
    p = sum(prior_flat != 0)
    # Total number of elements.
    num_elem = len(prior_flat)
    # True negative.
    tn = sum(np.logical_and(decision_flat == 0, prior_flat == 0))
    if p:
        sens = tp/p
    else:
        sens = np.nan
    if num_detect:
        acc = tp/num_detect
    else:
        acc = np.nan
    if num_elem > p:
        spec = tn/(num_elem-p)
    else:
        spec = np.nan
    return sens, acc, spec


def get_sas_list(graphml_file, adj_mat_file, thresholds,
                 self_edge=False):
    """Get sensitivity, accuracy and specificity lists.

    Args:
        graphml_file: A GraphML file of the reconstructed network
            with weighted directed edges.
        adj_mat_file: Adjacency matrix of the ground truth.
        thresholds: A list of thresholds used to calculate the
            performance measures.
        self_edge: Indicator of whether self-edges are allowed.
            Default is false; i.e., self-edges in the ground thuth
            are ignored and not counted toward the sensitivity,
            accuracy or specificity.

    Returns:
        A 3-tuple of lists of the sensitivity, accuracy and
            specificity corresponding to the thresholds, or NaN
            if any denominator is zero.
    """
    network = nx.read_graphml(graphml_file)
    adj_mat = np.loadtxt(adj_mat_file, delimiter=' ')
    num_genes_in_adj_mat = adj_mat.shape[0]
    num_genes = len(network.nodes())
    # Pad the adj matrix of extra genes with zeros.
    if num_genes > num_genes_in_adj_mat:
        adj_mat_big = np.zeros((num_genes, num_genes))
        adj_mat_big[
            :num_genes_in_adj_mat, :num_genes_in_adj_mat
            ] = adj_mat
        adj_mat = adj_mat_big
    sens_ls = []
    acc_ls = []
    spec_ls = []
    net_sign = nx.adjacency_matrix(
        network, weight='sign'
        ).toarray()
    net_weight = nx.adjacency_matrix(
        network, weight='weight'
        ).toarray()
    # Reconstructed network with signed weight.
    net_sign_wt = net_sign*net_weight
    # Ternary network of the ground truth with 1, -1 and 0.
    gt_tern = np.sign(adj_mat)
    for th in thresholds:
        net_tern = np.array(net_sign_wt, copy=True)
        net_tern[np.absolute(net_tern) < th] = 0
        net_tern = np.sign(net_tern)
        sens, acc, spec = get_sas(net_tern, gt_tern, self_edge)
        sens_ls.append(sens)
        acc_ls.append(acc)
        spec_ls.append(spec)
    return sens_ls, acc_ls, spec_ls


def plot_sas(graphml_file, gt, thresholds, self_edge=False,
             plots=False, output='output', metrics='all'):
    """Plot the accuracy-sensitivity curve and the sensiticity-
    specificity curve.

    Args:
        graphml_file: A dictionary with algorithm names as the key
            and GraphML format file paths as the value.
        gt: Adjacency matrix file of the ground truth.
        thresholds: A list of thresholds used to calculate the
            performance measures.
        self_edge: Indicator of whether self edges are allowed.
        plots: Indicator of whether the figures are to be plotted.
        output: Prefix for output figures.  An empty string
            indicates no output.
        metrics: Indicator for metrics.  Value can be a
                string or a list of strings.
            'all': 'auas', 'auss', 'best-f1' and 'vusas'.
            'none': Do not show metrics.
            metrics: A list of metrics.

    Returns:
        Plots or saves two figures for the AS curve and the SS
            curve.
    """
    algs = graphml_file.keys()
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    if metrics == 'none':
        metric_list = []
    elif metrics == 'all':
        metric_list = ['auas', 'auss', 'best-f1', 'vusas']
    else:
        metric_list = list(metrics)
    if metric_list:
        metric_display_names = {
            'auas': 'AUAS',
            'auss': 'AUSS',
            'best-f1': 'Best-F1',
            'vusas': 'VUSAS'
            }
        fig3, ax3 = plt.subplots()
        ind = np.arange(len(metric_list))
        width = 0.4/len(algs)
        rects = []
        offset = 0
    for alg in graphml_file:
        sas = get_sas_list(graphml_file[alg], gt,
                           thresholds, self_edge)
        ax1.plot(sas[0], sas[1], '-o', markerfacecolor='none')
        ax2.plot(sas[2], sas[0], '-o', markerfacecolor='none')
        if metric_list:
            metric_vals = []
            for metric in metric_list:
                metric_vals.append(get_metric(sas, metric))
            rects.append(ax3.bar(ind+offset*width, metric_vals,
                                 width))
            offset += 1
    if metric_list:
        ax3.set_ylabel('Metrics')
        ax3.set_title('Performance of network inference '
                      'algorithms')
        ax3.set_xticks(ind+width/2*(len(algs)-1))
        ax3.set_xticklabels([metric_display_names[m]
                             for m in metric_list])
        ax3.legend(tuple(r[0] for r in rects), algs)
    ax1.legend(algs)
    ax1.set_title('AS curve')
    ax1.set_xlabel('Sensitivity')
    ax1.set_ylabel('Accuracy')
    ax2.legend(algs)
    ax2.set_title('SS curve')
    ax2.set_xlabel('Specificity')
    ax2.set_ylabel('Sensitivity')
    if plots:
        fig1.show()
        fig2.show()
        if metric_list:
            fig3.show()
    if output:
        fig1.savefig(output+'-as.pdf')
        fig2.savefig(output+'-ss.pdf')
        if metric_list:
            fig3.savefig(output+'-metrics.pdf')
    return


def get_metric(sas, metric):
    """Get classification metric from sensitivity, accuracy and
    specificity.

    Get classification metrics using linear interpolation.

    Args:
        sas: A 3-tuple of lists of sensitivity, accuracy and
            specificity.  We assume NaNs are at the end of
            accuracy, if any.  The sensitivity must be
            non-increasing and specificity must be non-decreasing.
        metric: One of the four metrics.
            'auas': Area under the accuracy-sensitivity curve.
            'auss': Area under the sensitivity-specificity curve.
            'best-f1': Best harmonic mean of accuracy and
                sensitivity.
            'vusas': Volume under the sensitivity-accuracy-
                specificity curve.

    Returns:
        The value for the selected metric.
    """
    assert(len(sas[0]) == len(sas[1]) == len(sas[2]))
    sens = np.array(sas[0])
    acc = np.array(sas[1])
    spec = np.array(sas[2])
    if (
        # No data points at all.
        not len(sens) or
        # No positives so sensitivity is not defined.
        np.isnan(sens).any() or
        # Metric involves accuracy, and it is not defined at any
        # data point.
        metric in ['auas', 'vusas', 'best-f1'] and
            np.isnan(acc).all() or
        # Metric involves specificity, and it is not defined.
        metric in ['auss', 'vusas'] and np.isnan(spec).any()
        ):
        # Metric is not defined.
        return np.nan
    if metric == 'best-f1':
        # Harmonic mean with possibly NaNs.
        f1 = hmean(acc, sens)
        return max(f1[np.isfinite(f1)])
    # Add a 0-sensitivity point at the end for integration with
    # linear interpolation.
    acc = np.append(acc, acc[-1])
    sens = np.append(sens, 0)
    spec = np.append(spec, spec[-1])
    if metric == 'auas':
        # The integration is nonpositive because the sensitivity
        # sequence is assumed to be nonincreasing.
        return -integrate_lin(sens, acc)
    if metric == 'auss':
        return -integrate_lin(sens, spec)
    if metric == 'vusas':
        return -integrate_lin(sens, acc*spec)
    print('Unknown metric.')
    sys.exit(1)
    return


def hmean(a, b):
    """Calculate the entrywise harmonic means of two arrays.

    Args:
        a: An array of numbers or NaNs.
        b: Another array of numbers or NaNs.

    Returns:
        An array of the entrywise harmonic means of a and b.
        Returns NaN for NaN or negative input.
    """
    c = []
    for i in range(len(a)):
        if (np.isnan(a[i]) or np.isnan(b[i]) or
            a[i] < 0 or b[i] < 0):
            c.append(np.nan)
        elif not a[i] or not b[i]:
            c.append(0)
        else:
            c.append(2/(1/a[i]+1/b[i]))
    return np.array(c)


def integrate_lin(x, y):
    """Integration with linear interpolation.

    Args:
        x: An array of monotone numbers as the integration
            variable.
        y: An array of numbers as the function.  Note y could
            contain NaNs, which are ignored.

    Returns:
        The integral with linear interpolation.
    """
    x_finite = x[np.isfinite(y)]
    y_finite = y[np.isfinite(y)]
    return np.inner(
        x_finite[1:]-x_finite[:-1], (y_finite[1:]+y_finite[:-1])/2
        )
