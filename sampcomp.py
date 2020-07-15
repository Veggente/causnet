"""Calculate sample complexity of network reconstruction"""
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class NetworkHypothesisTesting:
    def __init__(self):
        self.hypotheses = [
            np.array([[0, 0.1], [0, 0]]),
            np.array([[0, -0.1], [0, 0]]),
        ]
        self.sigma_en_sq = 1
        self.sigma_in_sq = 1
        self.sigma_te_sq = 1
        self.prob_error = 0.05
        self.one_shot = True
        self.samp_times = 2
        self.num_rep = 1

    def bhatta_bound(self):
        """Calculate Bhattacharyya bounds"""
        cov_mat_0 = self.gen_cov_mat(
            self.hypotheses[0],
            self.sigma_in_sq,
            self.sigma_en_sq,
            self.samp_times,
            self.num_rep,
            self.one_shot,
            self.sigma_te_sq,
        )
        cov_mat_1 = self.gen_cov_mat(
            self.hypotheses[1],
            self.sigma_in_sq,
            self.sigma_en_sq,
            self.samp_times,
            self.num_rep,
            self.one_shot,
            self.sigma_te_sq,
        )
        rho = bhatta_coeff(cov_mat_0, cov_mat_1)
        return (
            1 / 2 * np.log(1 - (1 - 2 * self.prob_error) ** 2) / np.log(rho),
            np.log(2 * self.prob_error) / np.log(rho),
        )

    def gen_cov_mat(
        self,
        adj_mat,
        sigma_in_sq,
        sigma_en_sq,
        num_time,
        num_rep,
        one_shot,
        sigma_te_sq,
    ):
        """Generate covariance matrix for the observations of possibly multiple genes under Gaussian linear model for a single condition.

        Args:
            adj_mat: array
                Network adjacency matrix.
            sigma_in_sq: float
                Individual variation.
            sigma_en_sq: float
                Environmental variation.
            num_time: int
                Number of sampling times.
            num_rep: int
                Number of replicates.
            one_shot: bool
                Indicator of one-shot sampling.
            sigma_te_sq: float
                Technical variation.

        Returns: array
            The covariance matrix.
        """
        num_genes = adj_mat.shape[0]
        num_samples = num_time * num_rep * num_genes
        cov_mat = np.empty((num_samples, num_samples))
        for tidx1 in range(num_time):
            for tidx2 in range(num_time):
                for ridx1 in range(num_rep):
                    for ridx2 in range(num_rep):
                        cov_mat[
                            tidx1 * num_rep * num_genes
                            + ridx1 * num_genes : tidx1 * num_rep * num_genes
                            + (ridx1 + 1) * num_genes,
                            tidx2 * num_rep * num_genes
                            + ridx2 * num_genes : tidx2 * num_rep * num_genes
                            + (ridx2 + 1) * num_genes,
                        ] = cov_mat_small(
                            adj_mat,
                            sigma_in_sq,
                            sigma_en_sq,
                            sigma_te_sq,
                            tidx1,
                            tidx2,
                            ridx1,
                            ridx2,
                            one_shot,
                        )
        return cov_mat


def cov_mat_small(
    adj_mat, sigma_in_sq, sigma_en_sq, sigma_te_sq, tidx1, tidx2, ridx1, ridx2, one_shot
):
    """Calculate small covariance matrix"""
    num_genes = adj_mat.shape[0]
    if (tidx1, ridx1) == (tidx2, ridx2):
        return (sigma_in_sq + sigma_en_sq) * geom_sum_mat(
            adj_mat, tidx1 + 1, tidx2 + 1
        ) + sigma_te_sq * np.identity(num_genes)
    elif ridx1 == ridx2 and not one_shot:
        return (sigma_in_sq + sigma_en_sq) * geom_sum_mat(adj_mat, tidx1 + 1, tidx2 + 1)
    else:
        return sigma_en_sq * geom_sum_mat(adj_mat, tidx1 + 1, tidx2 + 1)


def geom_sum_mat(a, k1, k2):
    """Partial sum of the matrix geometric series.

    Args:
        a: array
            A square matrix.
        k1: int
            Maximum power on the left.
        k2: int
            Maximum power on the right.

    Returns: array
        \sum_{\tau = 1}^{k1\wedge k2}(a^T)**(k1-\tau)*a**(k2-\tau).
    """
    a_power = np.identity(a.shape[0])
    sum_mat = np.identity(a.shape[0])
    for i in range(min(k1, k2) - 1):
        a_power = a.T.dot(a_power).dot(a)
        sum_mat += a_power
    if k1 >= k2:
        return np.linalg.matrix_power(a.T, k1 - k2).dot(sum_mat)
    else:
        return sum_mat.dot(np.linalg.matrix_power(a, k2 - k1))


def bhatta_coeff(cov_mat_0, cov_mat_1):
    """Bhattacharyya coefficient"""
    return (np.linalg.det(cov_mat_0) * np.linalg.det(cov_mat_1)) ** (
        1 / 4
    ) / np.linalg.det((cov_mat_0 + cov_mat_1) / 2) ** (1 / 2)


def plot_bounds(sigma_te_sq=0, saveas="bhatta_bound.eps", start_delta=0.1, diagonal=0):
    """Plot Bhattacharyya bounds against regulation strength.

    Args:
        sigma_te_sq: float
            Technical variation.
        saveas: str
            Output file.
        start_delta: float
            Starting value for delta.
        diagonal: float
            Diagonal entries of the adjacency matrix.

    Returns: None
        Save plot to file.
    """
    ht = NetworkHypothesisTesting()
    ht.sigma_te_sq = sigma_te_sq
    for i in range(2):
        for j in range(2):
            ht.hypotheses[i][j, j] = diagonal
    lb = {one_shot: [] for one_shot in [True, False]}
    ub = {one_shot: [] for one_shot in [True, False]}
    delta_array = np.linspace(start_delta, 0.9, 100)
    for delta in delta_array:
        ht.hypotheses[0][0, 1] = delta
        ht.hypotheses[1][0, 1] = -delta
        for one_shot in [True, False]:
            ht.one_shot = one_shot
            lower, upper = ht.bhatta_bound()
            lb[one_shot].append(lower)
            ub[one_shot].append(upper)
    plt.figure()
    one_shot_str = lambda x: 'one-shot' if x else 'multi-shot'
    for one_shot in [True, False]:
        plt.plot(delta_array, lb[one_shot], label=one_shot_str(one_shot)+', lower bound')
        plt.plot(delta_array, ub[one_shot], label=one_shot_str(one_shot)+', upper bound')
    plt.legend()
    plt.xlabel(r"$\Delta$")
    plt.ylabel("sample complexity")
    plt.savefig(saveas)
