"""Calculate sample complexity of network reconstruction"""
from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_discrete

plt.style.use("ggplot")


class NetworkHypothesisTesting:  # pylint: disable=too-many-instance-attributes
    """Network hypothesis testing."""

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
        cov_mat_0 = gen_cov_mat(
            self.hypotheses[0],
            self.sigma_in_sq,
            self.sigma_en_sq,
            self.samp_times,
            self.num_rep,
            self.one_shot,
            self.sigma_te_sq,
        )
        cov_mat_1 = gen_cov_mat(
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

    def sim_er_genie_bhatta_lb(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        num_genes: int,
        prob_conn: float,
        spec_rad: float,
        num_sims: int,
        num_cond: int,
        bayes: bool = True,
    ) -> float:
        """Simulate genie-aided Bhattacharyya lower bound.

        Simulate ER graphs to get a genie-aided Bhattacharyya lower
        bound on average error probability.

        Args:
            num_genes: Number of genes/nodes.
            prob_conn: Probability of connection.
            spec_rad: The desired spectral radius.
            num_sims: Number of simulations.
            num_cond: Number of conditions.
            bayes: Use prob_conn as the Bayesian prior.

        Returns:
            Bhattacharyya lower bound.

        """
        lb_list = []
        if num_genes > 1:
            lb_cross_list = []
        if bayes:
            prior = (1 - prob_conn, prob_conn)
        else:
            prior = (1 / 2, 1 / 2)
        for _ in range(num_sims):
            er_graph, weight = erdos_renyi(num_genes, prob_conn, spec_rad)
            # Autoregulation.  Genie tells everything except the self-edge (0, 0).
            auto_adj_mat = self.genie_hypotheses(er_graph, (0, 0), weight, spec_rad)
            auto_cov_mat_0 = gen_cov_mat(
                auto_adj_mat[0],
                self.sigma_in_sq,
                self.sigma_en_sq,
                self.samp_times,
                self.num_rep,
                self.one_shot,
                self.sigma_te_sq,
            )
            auto_cov_mat_1 = gen_cov_mat(
                auto_adj_mat[1],
                self.sigma_in_sq,
                self.sigma_en_sq,
                self.samp_times,
                self.num_rep,
                self.one_shot,
                self.sigma_te_sq,
            )
            rho_auto = bhatta_coeff(auto_cov_mat_0, auto_cov_mat_1)
            lb_list.append(
                self.lower_bound_on_error_prob(rho_auto, num_cond, prior=prior)
            )
            if num_genes > 1:
                # Cross regulation.
                cross_adj_mat = self.genie_hypotheses(
                    er_graph, (0, 1), weight, spec_rad
                )
                cross_cov_mat_0 = gen_cov_mat(
                    cross_adj_mat[0],
                    self.sigma_in_sq,
                    self.sigma_en_sq,
                    self.samp_times,
                    self.num_rep,
                    self.one_shot,
                    self.sigma_te_sq,
                )
                cross_cov_mat_1 = gen_cov_mat(
                    cross_adj_mat[1],
                    self.sigma_in_sq,
                    self.sigma_en_sq,
                    self.samp_times,
                    self.num_rep,
                    self.one_shot,
                    self.sigma_te_sq,
                )
                rho_cross = bhatta_coeff(cross_cov_mat_0, cross_cov_mat_1)
                lb_cross_list.append(
                    self.lower_bound_on_error_prob(rho_cross, num_cond, prior=prior)
                )
        auto_lb_stat = np.mean(lb_list), np.std(lb_list)
        if num_genes == 1:
            return auto_lb_stat
        cross_lb_stat = np.mean(lb_cross_list), np.std(lb_cross_list)
        return auto_lb_stat, cross_lb_stat

    def genie_hypotheses(  # pylint: disable=no-self-use
        self, graph: np.ndarray, pos: Tuple[int, int], weight: float, spec_rad: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate genie-aided hypotheses.

        Args:
            graph: Adjacency matrix.
            pos: Position of the unknown edge.
            weight: Edge scale.
            spec_rad: Desired spectral radius.

        Returns:
            Two hypotheses.
        """
        unknown_edge = graph[pos]
        # If weight is positive, adj_mat_0 has spectral radius exactly
        # spec_rad.  Then adj_mat_1 is different in one edge, so its
        # spectral radius might be different from spec_rad.  If weight
        # is zero, adj_mat_0 has zero spectral radius.  We set the
        # different edge to be either zero or plus/minus spec_rad.
        adj_mat_0 = graph
        adj_mat_1 = graph.copy()
        if unknown_edge:
            adj_mat_1[pos] = 0
        else:
            if weight:
                scale = weight
            else:
                scale = spec_rad
            rademacher = np.random.binomial(1, 0.5) * 2 - 1
            adj_mat_1[pos] = scale * rademacher
        return adj_mat_0, adj_mat_1

    def lower_bound_on_error_prob(  # pylint: disable=no-self-use
        self, rho: float, num_cond: int, prior: Tuple[float, float] = (0.5, 0.5)
    ) -> float:
        """Lower bound on average error probability.

        Args:
            rho: Bhattacharyya coefficient for a single condition.
            num_cond: Number of conditions.
            prior: Prior distribution of the hypotheses.

        Returns:
            Lower bound.
        """
        if prior == (0.5, 0.5):
            return 1 / 2 * (1 - np.sqrt(1 - rho ** (2 * num_cond)))
        return prior[0] * prior[1] * rho ** (2 * num_cond)


def cov_mat_small(  # pylint: disable=too-many-arguments
    adj_mat: np.ndarray,
    sigma_in_sq: np.ndarray,
    sigma_en_sq: np.ndarray,
    sigma_te_sq: np.ndarray,
    tidx1: int,
    tidx2: int,
    ridx1: int,
    ridx2: int,
    one_shot: bool,
    initial: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Calculates small covariance matrix.

    Args:
        adj_mat: Adjacency matrix.
        sigma_in_sq: Individual variance.
        sigma_en_sq: Environmental variance.
        sigma_te_sq: Technical variance.
        tidx1: Time index 1.
        tidx2: Time index 2.
        ridx1: Replicate index 1.
        ridx2: Replicate index 2.
        one_shot: One-shot sampling.
        initial: Initial covariance matrix for single replicate
            multi-shot sampling.

    Returns:
        Small covariance matrix.
    """
    num_genes = adj_mat.shape[0]
    if initial is not None:
        cov_mat = np.linalg.matrix_power(adj_mat.T, tidx1).dot(initial).dot(np.linalg.matrix_power(adj_mat, tidx2))
        times = (tidx1, tidx2)
    else:
        cov_mat = np.zeros(adj_mat.shape)
        times = (tidx1 + 1, tidx2 + 1)
    if (tidx1, ridx1) == (tidx2, ridx2):
        cov_mat += (sigma_in_sq + sigma_en_sq) * geom_sum_mat(
            adj_mat, *times
        ) + sigma_te_sq * np.identity(num_genes)
    elif ridx1 == ridx2 and not one_shot:
        cov_mat += (sigma_in_sq + sigma_en_sq) * geom_sum_mat(
            adj_mat, *times
        )
    else:
        cov_mat += sigma_en_sq * geom_sum_mat(adj_mat, *times)
    return cov_mat


def geom_sum_mat(
    matrix: np.ndarray, max_pow_1: int, max_pow_2: int, skip: bool = False
) -> np.ndarray:
    """Partial sum of the matrix geometric series.

    Args:
        matrix: A square matrix.
        max_pow_1: Maximum power on the left plus one.
        max_pow_2: Maximum power on the right plus one.
        skip: Skip the first term in the summation.

    Returns:
        sum_{tau = 1}^{max_pow_1 wedge max_pow_2}(a^T)**(max_pow_1-tau)*a**(max_pow_2-tau).
    """
    if not max_pow_1 or not max_pow_2:
        return np.zeros(matrix.shape)
    a_power = np.identity(matrix.shape[0])
    sum_mat = np.identity(matrix.shape[0])
    for i in range(min(max_pow_1, max_pow_2) - 1):
        a_power = matrix.T.dot(a_power).dot(matrix)
        if skip and i == 0:
            continue
        sum_mat += a_power
    if max_pow_1 >= max_pow_2:
        return np.linalg.matrix_power(matrix.T, max_pow_1 - max_pow_2).dot(sum_mat)
    return sum_mat.dot(np.linalg.matrix_power(matrix, max_pow_2 - max_pow_1))


def bhatta_coeff(cov_mat_0, cov_mat_1):
    """Bhattacharyya coefficient"""
    # Use np.linalg.slogdet to avoid overflow.
    logdet = [np.linalg.slogdet(cov_mat)[1] for cov_mat in [cov_mat_0, cov_mat_1]]
    logdet_avg = np.linalg.slogdet((cov_mat_0 + cov_mat_1) / 2)[1]
    return np.exp(sum(logdet) / 4 - logdet_avg / 2)


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
    hyp_test = NetworkHypothesisTesting()
    hyp_test.sigma_te_sq = sigma_te_sq
    for i in range(2):
        for j in range(2):
            hyp_test.hypotheses[i][j, j] = diagonal
    lower_bounds = {one_shot: [] for one_shot in [True, False]}
    upper_bounds = {one_shot: [] for one_shot in [True, False]}
    delta_array = np.linspace(start_delta, 0.9, 100)
    for delta in delta_array:
        hyp_test.hypotheses[0][0, 1] = delta
        hyp_test.hypotheses[1][0, 1] = -delta
        for one_shot in [True, False]:
            hyp_test.one_shot = one_shot
            lower, upper = hyp_test.bhatta_bound()
            lower_bounds[one_shot].append(lower)
            upper_bounds[one_shot].append(upper)
    plt.figure()
    one_shot_str = lambda x: "one-shot" if x else "multi-shot"
    for one_shot in [True, False]:
        plt.plot(
            delta_array,
            lower_bounds[one_shot],
            label=one_shot_str(one_shot) + ", lower bound",
        )
        plt.plot(
            delta_array,
            upper_bounds[one_shot],
            label=one_shot_str(one_shot) + ", upper bound",
        )
    plt.legend()
    plt.xlabel(r"$\Delta$")
    plt.ylabel("sample complexity")
    plt.savefig(saveas)


def gen_cov_mat(  # pylint: disable=too-many-arguments
    adj_mat: np.ndarray,
    sigma_in_sq: float,
    sigma_en_sq: float,
    num_time: int,
    num_rep: int,
    one_shot: bool,
    sigma_te_sq: float,
    skip: int = 0,
    initial: Optional[np.ndarray] = None,
):
    """Generate covariance matrix.

    Generate covariance matrix for the observations of possibly
    multiple genes under Gaussian linear model for a single condition.
    The initial condition (before time 0) is zero.

    Args:
        adj_mat: Network adjacency matrix.
        sigma_in_sq: Individual variation.
        sigma_en_sq: Environmental variation.
        num_time: Number of sampling times.
        num_rep: Number of replicates.
        one_shot: Indicator of one-shot sampling.
        sigma_te_sq: Technical variation.
        skip: Number of time slots skipped in subsampling.
        initial: Initial covariance matrix for single replicate.

    Returns: array
        The covariance matrix.

    """
    if initial is not None and (num_rep != 1 or one_shot):
        raise ValueError("Can only take initial covariance matrix for single replicate multi-shot sampling.")
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
                        tidx1 * (1 + skip),
                        tidx2 * (1 + skip),
                        ridx1,
                        ridx2,
                        one_shot,
                        initial,
                    )
    return cov_mat


def erdos_renyi(
    num_genes: int, prob_conn: float, spec_rad: float = 0.8
) -> Tuple[np.ndarray, float]:
    """Initialize an Erdos Renyi network as in Sun–Taylor–Bollt 2015.

    If the spectral radius is positive, the matrix is normalized
    to a spectral radius of spec_rad and the scale shows the
    normalization.  If the spectral radius is zero, the returned
    matrix will have entries of 0, 1, and -1, and the scale is set
    to zero.

    Args:
        num_genes: Number of genes/nodes.
        prob_conn: Probability of connection.
        spec_rad: The desired spectral radius.

    Returns:
        Adjacency matrix and its scale.
    """
    vals = np.asarray([-1.0, 0.0, 1.0])
    signed_edge_dist = rv_discrete(
        values=(np.arange(3), [prob_conn / 2, 1 - prob_conn, prob_conn / 2])
    )
    signed_edges = vals[signed_edge_dist.rvs(size=(num_genes, num_genes))]
    original_spec_rad = max(abs(np.linalg.eigvals(signed_edges)))
    if original_spec_rad:
        return signed_edges / original_spec_rad * spec_rad, spec_rad / original_spec_rad
    return signed_edges, 0

def asymptotic_cov_mat(initial: np.ndarray, adj_mat: np.ndarray, sigma_sq: float, num_iter: int) -> Tuple[np.ndarray, float]:
    """Gets the asymptotic covariance matrix iteratively.

    Args:
        initial: Initial covariance matrix.
        adj_mat: Adjacency matrix.
        sigma_sq: Total biological variance.
        num_iter: Number of iterations.

    Returns:
        Limiting covariance matrix and norm of the last difference.
    """
    last_cov_mat = initial
    for i in range(num_iter):
        new_cov_mat = adj_mat.T.dot(last_cov_mat).dot(adj_mat)+sigma_sq*np.identity(adj_mat.shape[0])
        if i == num_iter-1:
            difference = np.linalg.norm(new_cov_mat-last_cov_mat)
        last_cov_mat = new_cov_mat
    return last_cov_mat, difference
