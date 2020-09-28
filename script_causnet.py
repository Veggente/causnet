"""Script for CausNet performance evaluation."""
from typing import Tuple, Dict, Any, Callable, Optional
import inspect
import numpy as np
import matplotlib.pyplot as plt
import sampcomp
import causnet_bslr
from lasso import lasso_grn

plt.style.use("ggplot")


class Script:
    """Script for CausNet performance evaluation."""

    @staticmethod
    def recreate_stb_single(
        stationary: bool = True, num_genes: int = 200, num_times: int = 2000, lasso: Optional[float] = None, **kwargs
    ) -> Tuple[int, int, int, int]:
        """Recreates a single simulation in Sun–Taylor–Bollt.

        Args:
            stationary: Wait till process is stationary.
            num_genes: Number of genes.
            num_times: Number of times.
            lasso: Also use lasso with l1 regularizer coefficient.
            **spec_rad: float
                Spectral radius.
            **alpha: float
                Significance level for permutation test.
            **obs_noise: float
                Observation noise variance.

        Returns:
            Simulation errors.
        """
        adj_mat, _ = sampcomp.erdos_renyi(
            num_genes, 0.05, **filter_kwargs(kwargs, sampcomp.erdos_renyi)
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
        parents, signs = causnet_bslr.ocse(
            data_cell, 100, **filter_kwargs(kwargs, causnet_bslr.ocse)
        )
        full_network = np.zeros((num_genes, num_genes))
        for j in range(num_genes):
            for idx, i in enumerate(parents[j]):
                full_network[i, j] = signs[j][idx]
        if lasso is not None:
            parents_lasso, signs_lasso = lasso_grn(data_cell, lasso)
            full_network_lasso = np.zeros((num_genes, num_genes))
            for j in range(num_genes):
                for idx, i in enumerate(parents_lasso[j]):
                    full_network_lasso[i, j] = signs_lasso[j][idx]
            return get_errors(full_network, adj_mat), get_errors(full_network_lasso, adj_mat)
        return get_errors(full_network, adj_mat)

    def recreate_stb_multiple(self, lasso: Optional[float] = None, sims: int = 20, **kwargs) -> Tuple[float, float]:
        """Recreates error estimates in Sun–Taylor–Bollt.

        Args:
            lasso: lasso l1 regularizer coefficient.
            sim: Number of simulations.
            **spec_rad: float
                Spectral radius.
            **alpha: float
                Significance level for permutation test.
            **obs_noise: float
                Observation noise variance.
            **stationary: bool
                Wait till process is stationary.

        Returns:
            False negative ratio and false positive ratio.
        """
        false_neg = 0
        false_pos = 0
        negative = 0
        positive = 0
        if lasso is None:
            for _ in range(sims):
                new_fn, new_p, new_fp, new_n = self.recreate_stb_single(**kwargs)
                false_neg += new_fn
                false_pos += new_fp
                negative += new_n
                positive += new_p
            return false_neg / positive, false_pos / negative
        false_neg_lasso = 0
        false_pos_lasso = 0
        negative_lasso = 0
        positive_lasso = 0
        for _ in range(sims):
            res = self.recreate_stb_single(**kwargs, lasso=lasso)
            new_fn, new_p, new_fp, new_n = res[0]
            false_neg += new_fn
            false_pos += new_fp
            negative += new_n
            positive += new_p
            new_fn, new_p, new_fp, new_n = res[1]
            false_neg_lasso += new_fn
            false_pos_lasso += new_fp
            negative_lasso += new_n
            positive_lasso += new_p
        return false_neg / positive, false_pos / negative, false_neg_lasso / positive_lasso, false_pos_lasso / negative_lasso

    def recreate_plot_stb(self, saveas: str, lasso: Optional[float] = None, **kwargs) -> None:
        """Recreates error plots.

        Args:
            saveas: Path to save figure to.
            lasso: lasso l1 regularizer coefficient.
            **alpha: float
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

        Returns:
            Saves plot.
        """
        spec_rad_arr = np.linspace(0.1, 0.4, 7)
        errors = []
        for spec_rad in spec_rad_arr:
            errors.append(self.recreate_stb_multiple(lasso=lasso, spec_rad=spec_rad, **kwargs))
        errors = np.array(errors)
        np.savetxt(saveas + ".data", errors)
        plt.figure()
        plt.plot(spec_rad_arr, errors[:, 0], label="False negative ratio of oCSE")
        plt.plot(spec_rad_arr, errors[:, 1], label="False positive ratio of oCSE")
        if lasso:
            plt.plot(spec_rad_arr, errors[:, 2], label="False negative ratio of lasso-{}".format(lasso))
            plt.plot(spec_rad_arr, errors[:, 3], label="False positive ratio of lasso-{}".format(lasso))
        plt.xlabel("spectral radius")
        plt.ylabel("error")
        plt.legend()
        plt.savefig(saveas + "{}.eps".format(kwargs))


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
    <false negative> / <positive>
    and the false positive ratio is defined as
    <false positive> / <negative>
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
