#!/usr/bin/env python3
"""Quality check for weighted GRN reconstruction.

Functions:
    show_histograms: Save or plot edge visibility histograms.
    hist_w_correct: Save or plot edge visibility histograms with
        ground truth.
"""

import networkx as nx
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np


def show_histograms(graphml_file, ylim=0, self_edge=False,
                    output='', display=False, figsize=None,
                    show_mean=False):
    """Show histograms of the edge weights.

    Args:
        graphml_file: A dictionary of GraphML files.
        ylim: Y-axis limit.  0 means default limit.
        self_edge: Indicator for self-edges.
        output: Output filename.
        display: Indicates whether to display the plot.
        figsize: Figure size.
        show_mean: Indicator for showing the mean visibility
            with a black vertical line.

    Returns:
        Plots the histograms.
    """
    num_graphs = len(graphml_file)
    fig, ax = plt.subplots(num_graphs, sharey=True,
                           figsize=figsize)
    bins = np.linspace(0, 1, 101)
    width = 0.01
    for idx_gf, alg in enumerate(graphml_file):
        network = nx.read_graphml(graphml_file[alg])
        num_genes = len(network.nodes())
        if self_edge:
            total_num_edges = num_genes**2
        else:
            total_num_edges = num_genes*(num_genes-1)
        if self_edge:
            weights = [data['weight'] for u, v, data in
                       network.edges(data=True)]
        else:
            weights = [data['weight'] for u, v, data in
                       network.edges(data=True) if u != v]
        hist, _ = np.histogram(weights, bins=bins)
        # Assign to edges that never appear the weight of 0.
        hist[0] = total_num_edges-sum(hist)
        if num_graphs == 1:
            ax = [ax]
        ax[idx_gf].bar(bins[:-1]+width/2, hist/total_num_edges,
                       width)
        # ax[idx_gf].legend()
        ax[idx_gf].set_title(alg)
        if ylim:
            ax[idx_gf].set_ylim(0, ylim)
        if show_mean:
            ax[idx_gf].axvline(np.mean(weights), color='k')
    plt.tight_layout()
    if output:
        fig.savefig(output)
    if display:
        fig.show()
    return


def hist_w_correct(graphml_file, ground_truth, ylim=0,
                   self_edge=False, output='', display=False,
                   figsize=None, show_mean=False):
    """Histogram with correctness.

    Args:
        graphml_file: A dictionary of GraphML files.
        ground_truth: Adjacency matrix file for the ground truth.
        ylim: Y-axis limit.  0 means default limit.
        self_edge: Indicator for self-edges.
        output: Output filename.
        display: Indicates whether to display the plot.
        figsize: Figure size.
        show_mean: Indicator for showing the mean visibility with
            a black vertical line.

    Returns:
        Plots the histograms or saves as a file.
    """
    num_graphs = len(graphml_file)
    fig, ax = plt.subplots(num_graphs, sharey=True,
                           figsize=figsize)
    bins = np.linspace(0, 1, 101)
    width = 0.01
    adj_mat = np.loadtxt(ground_truth, delimiter=' ')
    for idx_gf, alg in enumerate(graphml_file):
        network = nx.read_graphml(graphml_file[alg])
        num_genes = len(network.nodes())
        if self_edge:
            total_num_edges = num_genes**2
        else:
            total_num_edges = num_genes*(num_genes-1)
        if self_edge:
            weights = [data['weight'] for u, v, data in
                       network.edges(data=True)]
            # Here we assume the gene ID for the ith gene is
            # 'Gene'+i, for 0 <= i <= n-1.
            weights_correct = [
                data['weight']
                for u, v, data in network.edges(data=True)
                if data['sign'] == np.sign(
                    adj_mat[int(u[4:]), int(v[4:])]
                    )
                ]
            false_positive = [
                True for u, v, data in network.edges(data=True)
                if not adj_mat[int(u[4:]), int(v[4:])]
                ]
        else:
            weights = [data['weight'] for u, v, data in
                       network.edges(data=True) if u != v]
            # Here we assume the gene ID for the ith gene is
            # 'Gene'+i, for 0 <= i <= n-1.
            weights_correct = [
                data['weight']
                for u, v, data in network.edges(data=True)
                if data['sign'] == np.sign(
                    adj_mat[int(u[4:]), int(v[4:])]
                    ) and u != v
                ]
            false_positive = [
                True for u, v, data in network.edges(data=True)
                if not adj_mat[int(u[4:]), int(v[4:])] and u != v
                ]
        hist, _ = np.histogram(weights, bins=bins)
        hist_correct, _ = np.histogram(weights_correct, bins=bins)
        # Assign to edges that never appear the weight of 0.
        hist[0] = total_num_edges-sum(hist)
        hist_correct[0] = (
            total_num_edges
            -np.sum(adj_mat != 0)
            -np.sum(false_positive)
            )
        if num_graphs == 1:
            ax = [ax]
        ax[idx_gf].bar(bins[:-1]+width/2, hist/total_num_edges,
                       width, label='False')
        ax[idx_gf].bar(
            bins[:-1]+width/2, hist_correct/total_num_edges,
            width, color='g', label='True'
            )
        ax[idx_gf].legend()
        if ylim:
            ax[idx_gf].set_ylim(0, ylim)
        if show_mean:
            ax[idx_gf].axvline(np.mean(weights), color='k')
        ax[idx_gf].set_title(alg)
    plt.tight_layout()
    if output:
        fig.savefig(output)
    if display:
        fig.show()
    return
