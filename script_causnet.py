"""Script for CausNet performance evaluation."""
from typing import Tuple, Dict, Any, Callable
import inspect
import numpy as np
import matplotlib.pyplot as plt
import sampcomp
import causnet_bslr

plt.style.use("ggplot")


class Script:
    """Script for CausNet performance evaluation."""

    def recreate_stb_single(  # pylint: disable=no-self-use
        self, **kwargs
    ) -> Tuple[int, int, int, int]:
        """Recreates a single simulation in Sun–Taylor–Bollt.

        Args:
            **spec_rad: float
                Spectral radius.
            **alpha: float
                Significance level for permutation test.

        Returns:
            Simulation errors.
        """
        num_genes = 200
        num_times = 2000
        adj_mat, _ = sampcomp.erdos_renyi(
            num_genes, 0.05, **filter_kwargs(kwargs, sampcomp.erdos_renyi)
        )
        data_cell = [gen_lin_gaussian(num_times * 10, adj_mat)[-num_times:, :]]
        parents, signs = causnet_bslr.ocse(
            data_cell, 100, **filter_kwargs(kwargs, causnet_bslr.ocse)
        )
        full_network = np.zeros((num_genes, num_genes))
        for j in range(num_genes):
            for idx, i in enumerate(parents[j]):
                full_network[i, j] = signs[j][idx]
        return get_errors(full_network, adj_mat)

    def recreate_stb_multiple(self, **kwargs) -> Tuple[float, float]:
        """Recreates error estimates in Sun–Taylor–Bollt.

        Args:
            **spec_rad: float
                Spectral radius.
            **alpha: float
                Significance level for permutation test.

        Returns:
            False negative ratio and false positive ratio.
        """
        false_neg = 0
        false_pos = 0
        negative = 0
        positive = 0
        for _ in range(20):
            new_fn, new_p, new_fp, new_n = self.recreate_stb_single(**kwargs)
            false_neg += new_fn
            false_pos += new_fp
            negative += new_n
            positive += new_p
        return false_neg / positive, false_pos / negative

    def recreate_plot_stb(self, alpha: float, saveas: str) -> None:
        """Recreates error plots.

        Args:
            alpha: Significance level for permutation test.
            saveas: Path to save figure to.

        Returns:
            Saves plot.
        """
        spec_rad_arr = np.linspace(0.1, 0.4, 7)
        errors = []
        for spec_rad in spec_rad_arr:
            errors.append(self.recreate_stb_multiple(spec_rad=spec_rad, alpha=alpha))
        errors = np.array(errors)
        np.savetxt(saveas + ".data", errors)
        plt.figure()
        plt.plot(spec_rad_arr, errors[:, 0], label="False negative ratio")
        plt.plot(spec_rad_arr, errors[:, 1], label="False positive ratio")
        plt.xlabel("spectral radius")
        plt.ylabel("error")
        plt.savefig(saveas)


def gen_lin_gaussian(num_times: int, adj_mat: np.ndarray) -> np.ndarray:
    """Generate linear Gaussian dynamics.

    Args:
        num_times: Number of times.
        adj_mat: Adjacency matrix.

    Returns:
        T-by-n array, where T and n are the numbers of times and genes.
    """
    num_genes = adj_mat.shape[0]
    data = np.empty((num_times, num_genes))
    driving_noise = np.random.randn(num_times, num_genes)
    data[0, :] = driving_noise[0, :]
    for i in range(1, num_times):
        data[i, :] = data[i - 1, :].dot(adj_mat) + driving_noise[i, :]
    return data


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
