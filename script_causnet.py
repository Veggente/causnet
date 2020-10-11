"""Script for CausNet performance evaluation."""
from typing import Tuple, Dict, Any, Callable, Optional, List
import inspect
import json
import numpy as np
import matplotlib.pyplot as plt
import sampcomp
import causnet_bslr
from lasso import lasso_grn

plt.style.use("ggplot")


class Script:
    """Script for CausNet performance evaluation."""

    @staticmethod
    def recreate_stb_single(  # pylint: disable=too-many-locals, too-many-arguments
        stationary: bool = True,
        num_genes: int = 200,
        prob_conn: float = 0.05,
        num_times: int = 2000,
        lasso: Optional[List[float]] = None,
        alpha: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, Dict[float, Dict[str, int]]]:
        """Recreates a single simulation in Sun–Taylor–Bollt.

        Args:
            stationary: Wait till process is stationary.
            num_genes: Number of genes.
            prob_conn: Probability of connection.
            num_times: Number of times.
            lasso: Also use lasso with l1 regularizer coefficient.
            alpha: Significance level for permutation test.
            **spec_rad: float
                Spectral radius.
            **obs_noise: float
                Observation noise variance.

        Returns:
            Performance counts.  E.g.,
                {"ocse": {0.05: {"fn": 10, "pos": 50, "fp": 3, "neg": 50}},
                 "lasso": {2.0: {"fn": 12, "pos": 50, "fp": 5, "neg": 50}}}
        """
        adj_mat, _ = sampcomp.erdos_renyi(
            num_genes, prob_conn, **filter_kwargs(kwargs, sampcomp.erdos_renyi)
        )
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
                fn, pos, fp, neg = get_errors(full_network, adj_mat)  # pylint: disable=invalid-name
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
                fn, pos, fp, neg = get_errors(full_network, adj_mat)  # pylint: disable=invalid-name
                count["lasso"][this_lasso]["fn"] = fn
                count["lasso"][this_lasso]["pos"] = pos
                count["lasso"][this_lasso]["fp"] = fp
                count["lasso"][this_lasso]["neg"] = neg
        return count

    def recreate_stb_multiple(
        self, sims: int = 20, **kwargs
    ) -> Dict[str, Dict[float, Dict[str, float]]]:
        """Recreates error estimates in Sun–Taylor–Bollt.

        Args:
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
            False negative ratios and false positive ratios.
        """
        count = self.recreate_stb_single(**kwargs)
        for _ in range(sims - 1):
            new_count = self.recreate_stb_single(**kwargs)
            for alg in count:
                for param in count[alg]:
                    for metric in count[alg][param]:
                        count[alg][param][metric] += new_count[alg][param][metric]
        res = {}
        for alg in count:
            res[alg] = {}
            for param in count[alg]:
                res[alg][param] = {
                    "fnr": count[alg][param]["fn"] / count[alg][param]["pos"],
                    "fpr": count[alg][param]["fp"] / count[alg][param]["neg"],
                }
        return res

    def recreate_plot_stb(
        self, saveas: str, spec_rad_arr: List[float], plot: bool = True, from_file: str = "", **kwargs
    ) -> None:
        """Recreates error plots.

        Args:
            saveas: Path to save figure to.
            spec_rad_arr: Spectral radius array.
            plot: Plots the figure.
            from_file: Load data from file.
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
                errors = json.load(f)
        else:
            errors = {}
            for spec_rad in spec_rad_arr:
                errors[spec_rad] = self.recreate_stb_multiple(spec_rad=spec_rad, **kwargs)
            with open(saveas + "-{}.data".format(kwargs_str), "w") as f:
                json.dump(errors, f, indent=4)
        if plot:
            self.plot_roc(errors, saveas + kwargs_str)

    @staticmethod
    def plot_roc(
        errors: Dict[float, Dict[str, Dict[float, Dict[str, float]]]], saveas: str
    ) -> None:
        """Plot ROC curves.

        Args:
            errors: False negative ratios and false positive ratios.
            saveas: Output prefix.

        Returns:
            Saves figures.
        """
        plt.figure()
        for spec_rad in errors:
            for alg in errors[spec_rad]:
                tpr = [
                    1 - errors[spec_rad][alg][param]["fnr"]
                    for param in errors[spec_rad][alg]
                ]
                fpr = [
                    errors[spec_rad][alg][param]["fpr"]
                    for param in errors[spec_rad][alg]
                ]
                plt.plot(fpr, tpr, "-o", label=alg + r", $\rho = $" + str(spec_rad))
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
