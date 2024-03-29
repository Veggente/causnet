"""Script for CausNet performance evaluation."""
from typing import Tuple, Dict, Any, Callable, Optional, List
import inspect
import json
from itertools import product
import importlib
import numpy as np
import matplotlib.pyplot as plt
import sampcomp
import causnet_bslr
from lasso import lasso_grn

importlib.import_module("net-inf-eval.network_bc")
plt.style.use("ggplot")


class Script:
    """Script for CausNet performance evaluation."""

    @staticmethod
    def recreate_stb_single(  # pylint: disable=too-many-locals, too-many-arguments, too-many-branches
        stationary: bool = True,
        num_genes: int = 200,
        prob_conn: float = 0.05,
        num_times: int = 2000,
        lasso: Optional[List[float]] = None,
        alpha: Optional[List[float]] = None,
        bhatta: bool = False,
        spec_rad: float = 0.8,
        **kwargs
    ) -> Tuple[Dict[str, Dict[float, Dict[str, int]]], Optional[Tuple[float, float]]]:
        """Recreates a single simulation in Sun–Taylor–Bollt.

        Args:
            stationary: Wait till process is stationary.
            num_genes: Number of genes.
            prob_conn: Probability of connection.
            num_times: Number of times.
            lasso: Also use lasso with l1 regularizer coefficient.
            alpha: Significance level for permutation test.
            bhatta: Calculates Bhattacharyya coefficients.
            spec_rad: Spectral radius.
            **obs_noise: float
                Observation noise variance.

        Returns:
            1. Performance counts.  E.g.,
                {"ocse": {0.05: {"fn": 10, "pos": 50, "fp": 3, "neg": 50}},
                 "lasso": {2.0: {"fn": 12, "pos": 50, "fp": 5, "neg": 50}}}
            2. Average Bhattacharyya coefficients squared for positive
                and negative perturbations.
        """
        adj_mat_ter = sampcomp.erdos_renyi_ternary(num_genes, prob_conn)
        if bhatta:
            bc_list = []
            for i, j in product(range(num_genes), repeat=2):
                if i <= j:
                    bc_list.append(
                        np.array(
                            network_bc.bc_4_perturbation(
                                adj_mat_ter,
                                (i, j),
                                spec_rad,
                                num_times - 1,
                                **filter_kwargs(kwargs, network_bc.bc_4_perturbation)
                            )
                        )
                        ** 2
                    )
            avg_bc = np.mean(bc_list, axis=0)
        else:
            avg_bc = None
        adj_mat, _ = sampcomp.scale_by_spec_rad(adj_mat_ter, rho=spec_rad)
        if stationary:
            total_num_times = num_times * 10
            sampled_time = slice(-num_times, None)
        else:
            total_num_times = num_times
            sampled_time = slice(None)
        data_cell = [
            gen_lin_gaussian(
                total_num_times, adj_mat, **filter_kwargs(kwargs, gen_lin_gaussian)
            )[sampled_time, :]
        ]
        count = {}
        if alpha is not None:
            count["ocse"] = {}
            for this_alpha in alpha:
                count["ocse"][this_alpha] = {}
                parents, signs = causnet_bslr.ocse(
                    data_cell,
                    100,
                    alpha=this_alpha,
                    **filter_kwargs(kwargs, causnet_bslr.ocse)
                )
                full_network = np.zeros((num_genes, num_genes))
                for j in range(num_genes):
                    for idx, i in enumerate(parents[j]):
                        full_network[i, j] = signs[j][idx]
                fn, pos, fp, neg = get_errors(  # pylint: disable=invalid-name
                    full_network, adj_mat
                )
                count["ocse"][this_alpha]["fn"] = fn
                count["ocse"][this_alpha]["pos"] = pos
                count["ocse"][this_alpha]["fp"] = fp
                count["ocse"][this_alpha]["neg"] = neg
        if lasso is not None:
            count["lasso"] = {}
            for this_lasso in lasso:
                count["lasso"][this_lasso] = {}
                parents, signs = lasso_grn(data_cell, this_lasso)
                full_network = np.zeros((num_genes, num_genes))
                for j in range(num_genes):
                    for idx, i in enumerate(parents[j]):
                        full_network[i, j] = signs[j][idx]
                fn, pos, fp, neg = get_errors(  # pylint: disable=invalid-name
                    full_network, adj_mat
                )
                count["lasso"][this_lasso]["fn"] = fn
                count["lasso"][this_lasso]["pos"] = pos
                count["lasso"][this_lasso]["fp"] = fp
                count["lasso"][this_lasso]["neg"] = neg
        return count, avg_bc

    def recreate_stb_multiple(
        self, bhatta: bool, sims: int = 20, **kwargs
    ) -> Tuple[Dict[str, Dict[float, Dict[str, float]]], Optional[Tuple[float, float]]]:
        """Recreates error estimates in Sun–Taylor–Bollt.

        Args:
            bhatta: Calculates Bhattacharyya coefficients.
            sims: Number of simulations.
            **num_genes: int
                Number of genes.
            **prob_conn: float
                Probability of connection.
            **num_times: int
                Number of times.
            **lasso: Optional[List[float]]
                lasso with l1 regularizer coefficient.
            **spec_rad: float
                Spectral radius.
            **alpha: Optional[List[float]]
                Significance level for permutation test.
            **obs_noise: float
                Observation noise variance.
            **stationary: bool
                Wait till process is stationary.

        Returns:
            False negative ratios and false positive ratios, and
            optionally the Bhattacharyya coefficients squared.
        """
        count, avg_bc = self.recreate_stb_single(bhatta=bhatta, **kwargs)
        avg_bc_list = [avg_bc]
        for _ in range(sims - 1):
            new_count, new_avg_bc = self.recreate_stb_single(bhatta=bhatta, **kwargs)
            for alg in count:
                for param in count[alg]:
                    for metric in count[alg][param]:
                        count[alg][param][metric] += new_count[alg][param][metric]
            avg_bc_list.append(new_avg_bc)
        res = {}
        for alg in count:
            res[alg] = {}
            for param in count[alg]:
                res[alg][param] = {
                    "fnr": count[alg][param]["fn"] / count[alg][param]["pos"],
                    "fpr": count[alg][param]["fp"] / count[alg][param]["neg"],
                }
        if bhatta:
            return res, np.mean(avg_bc_list, axis=0)
        return res, None

    def recreate_plot_stb(  # pylint: disable=too-many-arguments, too-many-branches
        self,
        saveas: str,
        spec_rad_arr: List[float],
        plot: bool = True,
        from_file: str = "",
        bhatta: bool = False,
        **kwargs
    ) -> None:
        """Recreates error plots.

        Args:
            saveas: Path to save figure to.
            spec_rad_arr: Spectral radius array.
            plot: Plots the figure.
            from_file: Load data from file.
            bhatta: Calculates Bhattacharyya coefficients.
            **lasso: Optional[List[float]]
                lasso l1 regularizer coefficient.
            **alpha: Optional[List[float]]
                Significance level for permutation test.
            **obs_noise: float
                Observation noise variance.
            **stationary: bool
                Wait till process is stationary.
            **num_genes: int
                Number of genes.
            **num_times: int
                Number of times.
            **sims: int
                Number of simulations.
            **prob_conn: float
                Probability of connection.

        Returns:
            Saves plot and/or data to files.
        """
        kwargs_str = "-".join(
            [
                key + "_" + str(kwargs[key])
                for key in kwargs
                if key not in ["lasso", "alpha"]
            ]
        )
        if from_file:
            with open(from_file) as f:
                if bhatta:
                    errors, bc2_dict = json.load(f)
                else:
                    errors = json.load(f)
        else:
            errors = {}
            if bhatta:
                bc2_dict = {}
            for spec_rad in spec_rad_arr:
                if bhatta:
                    errors[spec_rad], bc2_dict[spec_rad] = self.recreate_stb_multiple(
                        spec_rad=spec_rad, bhatta=bhatta, **kwargs
                    )
                else:
                    errors[spec_rad], _ = self.recreate_stb_multiple(
                        spec_rad=spec_rad, bhatta=bhatta, **kwargs
                    )
            print(bc2_dict)
            bc2_dict = {key: np.mean(value) for key, value in bc2_dict.items()}
            with open(saveas + "-{}.data".format(kwargs_str), "w") as f:
                if bhatta:
                    save_vars = (errors, bc2_dict)
                else:
                    save_vars = errors
                json.dump(save_vars, f, indent=4)
        if plot:
            if bhatta:
                self.plot_roc(errors, saveas + "-" + kwargs_str, bc2_dict)
            else:
                self.plot_roc(errors, saveas + "-" + kwargs_str)

    @staticmethod
    def plot_roc(
        errors: Dict[float, Dict[str, Dict[float, Dict[str, float]]]],
        saveas: str,
        bc2_dict: Optional[Dict[float, float]] = None,
    ) -> None:
        """Plot ROC curves.

        Args:
            errors: False negative ratios and false positive ratios.
            saveas: Output prefix.
            bc2_dict: Bhattacharyya coefficient squared.

        Returns:
            Saves figures.
        """
        plt.figure()
        for idx, spec_rad in enumerate(errors):
            for alg in errors[spec_rad]:
                tpr = [
                    1 - errors[spec_rad][alg][param]["fnr"]
                    for param in errors[spec_rad][alg]
                ]
                fpr = [
                    errors[spec_rad][alg][param]["fpr"]
                    for param in errors[spec_rad][alg]
                ]
                if alg == "ocse":
                    symbol = "-o"
                elif alg == "lasso":
                    symbol = "--x"
                else:
                    raise ValueError("Unknown algorithm.")
                plt.plot(
                    fpr,
                    tpr,
                    symbol,
                    color="C{}".format(idx),
                    label=alg + r", $\rho = $" + str(spec_rad),
                )
            if bc2_dict:
                fpr_full = np.linspace(0, 1)
                bc2_fpr_diff = np.sqrt(bc2_dict[spec_rad]) - np.sqrt(fpr_full)
                abs_bound = 1 - (bc2_fpr_diff * (bc2_fpr_diff >= 0)) ** 2
                plt.plot(
                    fpr_full,
                    abs_bound,
                    "-.",
                    color="C{}".format(idx),
                    label=r"BC bound, $\rho = $" + str(spec_rad),
                )
        plt.legend(loc="best")
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.xlim(-0.02, 1.02)
        plt.ylim(-0.02, 1.02)
        plt.savefig(saveas + ".eps")


def gen_lin_gaussian(
    num_times: int, adj_mat: np.ndarray, obs_noise: float = 0.0
) -> np.ndarray:
    """Generate linear Gaussian dynamics.

    Args:
        num_times: Number of times.
        adj_mat: Adjacency matrix.
        obs_noise: Observation noise variance.

    Returns:
        T-by-n array, where T and n are the numbers of times and genes.
    """
    num_genes = adj_mat.shape[0]
    data = np.empty((num_times, num_genes))
    driving_noise = np.random.randn(num_times, num_genes)
    data[0, :] = driving_noise[0, :]
    for i in range(1, num_times):
        data[i, :] = data[i - 1, :].dot(adj_mat) + driving_noise[i, :]
    return data + np.sqrt(obs_noise) * np.random.randn(num_times, num_genes)


def get_errors(decision: np.ndarray, truth: np.ndarray) -> Tuple[int, int, int, int]:
    """Get inference errors.

    The false negative ratio is defined as
    <false negative> / <condition positive>
    and the false positive ratio is defined as
    <false positive> / <condition negative>
    according to Sun–Taylor–Bollt.

    Args:
        decision: Decision array.
        truth: Ground truth array.

    Returns:
        False negative, positive, false positive, and negative.
    """
    fn_counter = 0
    fp_counter = 0
    for i in range(decision.shape[0]):
        for j in range(decision.shape[1]):
            if truth[i, j] and not decision[i, j]:
                fn_counter += 1
            if not truth[i, j] and decision[i, j]:
                fp_counter += 1
    positive = int(np.sum(abs(np.sign(truth))))
    negative = int(np.multiply(*decision.shape)) - positive
    return fn_counter, positive, fp_counter, negative


def filter_kwargs(kwargs: Dict[str, Any], func: Callable) -> Dict[str, Any]:
    """Filter keyword arguments for a function."""
    return {
        key: value
        for key, value in kwargs.items()
        if key in inspect.getfullargspec(func).args
    }
