"""Script for sampcomp."""
from typing import Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
import sampcomp
from script_causnet import filter_kwargs

plt.style.use("ggplot")


def script_plot_1():
    """Single edge with autoregulation."""
    diagonal = 0.8
    sampcomp.plot_bounds(
        saveas="bhatta-bound-a{}-s0-d0.1.eps".format(diagonal), diagonal=diagonal,
    )
    sampcomp.plot_bounds(
        saveas="bhatta-bound-a{}-s0-d0.5.eps".format(diagonal),
        start_delta=0.5,
        diagonal=diagonal,
    )
    sampcomp.plot_bounds(
        saveas="bhatta-bound-a{}-s1-d0.5.eps".format(diagonal),
        start_delta=0.5,
        sigma_te_sq=1,
        diagonal=diagonal,
    )
    sampcomp.plot_bounds(
        saveas="bhatta-bound-a{}-s1-d0.1.eps".format(diagonal),
        sigma_te_sq=1,
        diagonal=diagonal,
    )


def script_plot_2():
    """Plot noisy bounds for single edge."""
    sampcomp.plot_bounds(saveas="bhatta-bound-s0-d0.1.eps")
    sampcomp.plot_bounds(saveas="bhatta-bound-s0-d0.5.eps", start_delta=0.5)
    sampcomp.plot_bounds(
        saveas="bhatta-bound-s1-d0.5.eps", start_delta=0.5, sigma_te_sq=1
    )
    sampcomp.plot_bounds(saveas="bhatta-bound-s1-d0.1.eps", sigma_te_sq=1)


def plot_lb_w_time(num_sims: int = 20) -> None:
    """Plot lower bounds with sampling times.

    Args:
        num_sims: Number of simulations.

    Returns:
        Save plot.
    """
    sim_net = sampcomp.NetworkHypothesisTesting()
    sim_net.sigma_in_sq = 0
    sim_net.sigma_te_sq = 0
    num_genes = 20
    prob_conn = 0.5
    spec_rad = 0.8
    time_list = [2, 4, 5, 8, 10, 20, 25, 40]
    lower_bounds = []
    for samp_time in time_list:
        num_cond = 200 / samp_time
        sim_net.samp_times = samp_time
        lower_bounds.append(
            sim_net.sim_er_genie_bhatta_lb(
                num_genes, prob_conn, spec_rad, num_sims, num_cond
            )[1]
        )
    lower_bounds = np.asarray(lower_bounds)
    plt.figure()
    plt.errorbar(time_list, lower_bounds[:, 0], yerr=lower_bounds[:, 1])
    plt.xlabel("number of sampling times")
    plt.ylabel("lower bound on average error probability")
    plt.savefig(
        "/Users/veggente/Documents/workspace/python/sampcomp/pe_v_time_n{}.eps".format(
            num_sims
        )
    )


def plot_lb_w_network_size(num_sims: int = 20, sigma_te_sq: float = 0) -> None:
    """Plot lower bounds with network size.

    Args:
        num_sims: Number of simulations.
        sigma_te_sq: Technical variation.

    Returns:
        Save plot.
    """
    sim_net = sampcomp.NetworkHypothesisTesting()
    sim_net.sigma_in_sq = 0
    sim_net.sigma_te_sq = sigma_te_sq
    num_genes_list = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    prob_conn = [10 / num_genes for num_genes in num_genes_list]
    spec_rad = 0.8
    samp_time = 10
    num_cond = 20
    lower_bounds = []
    for counter, num_genes in enumerate(num_genes_list):
        sim_net.samp_times = samp_time
        lower_bounds.append(
            sim_net.sim_er_genie_bhatta_lb(
                num_genes, prob_conn[counter], spec_rad, num_sims, num_cond
            )[1]
        )
    lower_bounds = np.asarray(lower_bounds)
    plt.figure()
    plt.errorbar(num_genes_list, lower_bounds[:, 0], yerr=lower_bounds[:, 1])
    plt.xlabel("network size")
    plt.ylabel("lower bound on average error probability")
    plt.savefig(
        "/Users/veggente/Documents/workspace/python/sampcomp/pe_v_network_size{}_n{}_t{}.eps".format(  # pylint: disable=line-too-long
            num_genes_list[-1], num_sims, sim_net.sigma_te_sq
        )
    )


def plot_lb_w_spec_rad(num_sims: int = 20, sigma_te_sq: float = 0) -> None:
    """Plot lower bounds with spectral radius.

    Args:
        num_sims: Number of simulations.
        sigma_te_sq: Technical variation.

    Returns:
        Save plot.
    """
    sim_net = sampcomp.NetworkHypothesisTesting()
    sim_net.sigma_in_sq = 0
    sim_net.sigma_te_sq = sigma_te_sq
    num_genes = 200
    prob_conn = 10 / num_genes
    spec_rad_list = [0.1, 0.2, 0.4, 0.8]
    samp_time = 10
    num_cond = 200
    lower_bounds = []
    sim_net.samp_times = samp_time
    for spec_rad in spec_rad_list:
        lower_bounds.append(
            sim_net.sim_er_genie_bhatta_lb(
                num_genes, prob_conn, spec_rad, num_sims, num_cond
            )[1]
        )
    lower_bounds = np.asarray(lower_bounds)
    plt.figure()
    plt.errorbar(spec_rad_list, lower_bounds[:, 0], yerr=lower_bounds[:, 1])
    plt.xlabel("spectral radius")
    plt.ylabel("lower bound on average error probability")
    plt.savefig(
        "/Users/veggente/Documents/workspace/python/sampcomp/pe_v_spec_rad_n{}_c{}_t{}.eps".format(
            num_sims, num_cond, sim_net.sigma_te_sq
        )
    )


def plot_sub_samp(seed: int = 0):
    """Plot subsampling results."""
    np.random.seed(seed)
    er_graph, weight = sampcomp.erdos_renyi(10, 0.2, 0.8)
    network_ht = sampcomp.NetworkHypothesisTesting()
    adj_mat_pair = list(network_ht.genie_hypotheses(er_graph, (0, 1), weight, 0.8))
    bhatta_list = []
    for skip in range(30):
        cov_mat = [
            sampcomp.gen_cov_mat(adj_mat, 0, 1, 10, 1, False, 1, skip)
            for adj_mat in adj_mat_pair
        ]
        bhatta_list.append(sampcomp.bhatta_coeff(*cov_mat))
    print(adj_mat_pair)
    plt.figure()
    plt.plot(np.arange(1, 31), bhatta_list)
    plt.xlabel("sampling interval T")
    plt.ylabel("Bhattacharyya coefficient")
    plt.savefig("subsampling_s{}.eps".format(seed))


def stb_lb(
    spec_rad: float,
    samp_times: int,
    sigma_te_sq: float,
    and_upper: bool = False,
    **kwargs
) -> Union[float, Tuple[float, float]]:
    """Calculates lower bound on average error rate.

    Average is on the cross edges.

    Args:
        spec_rad: Spectral radius.
        samp_times: Number of sampling times.
        sigma_te_sq: Technical variance.
        and_upper: Also compute upper bounds.
        **stationary: bool
            Start from stationary distribution.

    Returns:
        Bhattacharyya lower bound (or lower and upper bounds) on average error rate.
    """
    num_genes = 200
    sigma_en_sq = 1
    sigma_in_sq = 0
    num_rep = 1
    prob_conn = 0.05
    num_sims = 10
    num_cond = 2000 / samp_times
    network_ht = sampcomp.NetworkHypothesisTesting()
    network_ht.sigma_en_sq = sigma_en_sq
    network_ht.sigma_in_sq = sigma_in_sq
    network_ht.sigma_te_sq = sigma_te_sq
    network_ht.samp_times = samp_times
    network_ht.num_rep = num_rep
    network_ht.one_shot = False
    sim_lower_bound = network_ht.sim_er_genie_bhatta_lb(
        num_genes,
        prob_conn,
        spec_rad,
        num_sims,
        num_cond,
        bayes=False,
        and_upper=and_upper,
        **filter_kwargs(kwargs, network_ht.sim_er_genie_bhatta_lb)
    )
    if and_upper:
        return sim_lower_bound[1][0], sim_lower_bound[3][0]
    return sim_lower_bound[1][0]


def plot_stb_lb(
    samp_times: int, saveas: str, sigma_te_sq: float, and_upper: bool = False, **kwargs
) -> None:
    """Plots Sun–Taylor–Bollt error lower bound.

    Args:
        samp_times: Number of sampling times.
        saveas: Path to save figure to.
        sigma_te_sq: Technical variance.
        and_upper: Also computes upper bounds.
        **stationary: bool
            Start from stationary distribution.

    Returns:
        Saves plot.
    """
    spec_rad_arr = np.linspace(0.1, 0.4, 7)
    err_lb = []
    for spec_rad in spec_rad_arr:
        err_lb.append(
            stb_lb(spec_rad, samp_times, sigma_te_sq, and_upper=and_upper, **kwargs)
        )
    err_lb = np.array(err_lb)
    np.savetxt(saveas + ".data", err_lb)
    if not and_upper:
        plt.figure()
        plt.plot(spec_rad_arr, err_lb)
        plt.xlabel("spectral radius")
        plt.ylabel("lower bound on average error")
        plt.savefig(saveas)


def plot_stb_lb_w_sim():
    """Plots Sun–Taylor–Bollt error with lower bound.

    Run after plot_stb_lb() and script_causnet.Script.recreate_plot_stb().
    """
    errors = np.loadtxt(
        "/Users/veggente/Documents/workspace/python/sampcomp/stb-sim-a0.02-n0.eps.data"
    )
    bounds = np.loadtxt(
        "/Users/veggente/Documents/workspace/python/sampcomp/stb-lb-s10-n0-stT-uT.data"
    )
    spec_rad_arr = np.linspace(0.1, 0.4, 7)
    plt.figure()
    plt.plot(spec_rad_arr, errors.mean(axis=1), label="average error rate")
    plt.plot(spec_rad_arr, bounds[:, 0], label="Bhattacharyya lower bound")
    plt.plot(spec_rad_arr, bounds[:, 1], "--", label="Bhattacharyya upper bound")
    plt.xlabel("spectral radius")
    plt.legend()
    plt.savefig(
        "/Users/veggente/Documents/workspace/python/sampcomp/stb-spec-rad-n0-08-14-2.eps"
    )


def plot_bhatta_w_time(num_sims: int = 20) -> None:
    """Plot Bhattacharyya coefficient with sampling times.

    Args:
        num_sims: Number of simulations.

    Returns:
        Save plot.
    """
    sim_net = sampcomp.NetworkHypothesisTesting()
    sim_net.sigma_in_sq = 0
    sim_net.sigma_te_sq = 0
    sim_net.one_shot = False
    num_genes = 200
    prob_conn = 0.05
    spec_rad = 0.8
    time_list = [2, 4, 6, 8, 10]
    bhatta_list = []
    for samp_time in time_list:
        sim_net.samp_times = samp_time
        bhatta_list.append(
            sim_net.sim_er_bhatta(
                num_genes, prob_conn, spec_rad, num_sims, stationary=True
            )[0]
        )
    plt.figure()
    plt.semilogy(time_list, bhatta_list)
    plt.xlabel("number of sampling times")
    plt.ylabel("Bhattacharyya coefficient")
    plt.savefig(
        "/Users/veggente/Documents/workspace/python/sampcomp/bhatta_v_time_n{}.eps".format(
            num_sims
        )
    )


def plot_bhatta_w_samp_rate(num_sims: int = 20) -> None:
    """Plot Bhattacharyya coefficient with sampling rate.

    Args:
        num_sims: Number of simulations.

    Returns:
        Save plot.
    """
    sim_net = sampcomp.NetworkHypothesisTesting()
    sim_net.sigma_in_sq = 0
    sim_net.sigma_te_sq = 0
    sim_net.one_shot = False
    num_genes = 200
    prob_conn = 0.05
    spec_rad = 0.8
    rate_list = [1, 2, 3, 4, 6]
    bhatta_list = []
    for rate in rate_list:
        sim_net.samp_times = int(12 / rate)
        bhatta_list.append(
            sim_net.sim_er_bhatta(
                num_genes, prob_conn, spec_rad, num_sims, stationary=True, skip=rate - 1
            )[0]
        )
    plt.figure()
    plt.plot(rate_list, bhatta_list)
    plt.xlabel("sampling rate")
    plt.ylabel("Bhattacharyya coefficient")
    plt.savefig(
        "/Users/veggente/Documents/workspace/python/sampcomp/bhatta_v_rate_n{}.eps".format(
            num_sims
        )
    )
