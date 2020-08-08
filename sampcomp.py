"""Calculate sample complexity of network reconstruction"""
import numpy as np
from typing import Tuple
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

    def sim_er_genie_bhatta_lb(
        self,
        num_genes: int,
        prob_conn: float,
        spec_rad: float,
        num_sims: int,
        num_cond: int,
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

        Returns:
            Bhattacharyya lower and upper bounds.

        """
        lb_list = []
        if num_genes > 1:
            lb_cross_list = []
        for i in range(num_sims):
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
                self.lower_bound_on_error_prob(
                    rho_auto, num_cond, prior=(1 - prob_conn, prob_conn)
                )
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
                    self.lower_bound_on_error_prob(
                        rho_cross, num_cond, prior=(1 - prob_conn, prob_conn)
                    )
                )
        auto_lb_stat = np.mean(lb_list), np.std(lb_list)
        if num_genes == 1:
            return auto_lb_stat
        else:
            cross_lb_stat = np.mean(lb_cross_list), np.std(lb_cross_list)
            return auto_lb_stat, cross_lb_stat

    def genie_hypotheses(
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

    def lower_bound_on_error_prob(
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
        else:
            return prior[0] * prior[1] * rho ** (2 * num_cond)


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
        sum_{tau = 1}^{k1 wedge k2}(a^T)**(k1-tau)*a**(k2-tau).
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
    one_shot_str = lambda x: "one-shot" if x else "multi-shot"
    for one_shot in [True, False]:
        plt.plot(
            delta_array, lb[one_shot], label=one_shot_str(one_shot) + ", lower bound"
        )
        plt.plot(
            delta_array, ub[one_shot], label=one_shot_str(one_shot) + ", upper bound"
        )
    plt.legend()
    plt.xlabel(r"$\Delta$")
    plt.ylabel("sample complexity")
    plt.savefig(saveas)

def gen_cov_mat(
    adj_mat: np.ndarray,
    sigma_in_sq: float,
    sigma_en_sq: float,
    num_time: int,
    num_rep: int,
    one_shot: bool,
    sigma_te_sq: float,
    skip: int = 0
):
    """Generate covariance matrix.

    Generate covariance matrix for the observations of possibly
    multiple genes under Gaussian linear model for a single
    condition.

    Args:
        adj_mat: Network adjacency matrix.
        sigma_in_sq: Individual variation.
        sigma_en_sq: Environmental variation.
        num_time: Number of sampling times.
        num_rep: Number of replicates.
        one_shot: Indicator of one-shot sampling.
        sigma_te_sq: Technical variation.
        skip: Number of time slots skipped in subsampling.

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
                        tidx1 * (1+skip),
                        tidx2 * (1+skip),
                        ridx1,
                        ridx2,
                        one_shot,
                    )
    return cov_mat

def erdos_renyi(
    num_genes: int, prob_conn: float, rho: float
) -> Tuple[np.ndarray, float]:
    """Initialize an Erdos Renyi network as in Sun–Taylor–Bollt 2015.

    If the spectral radius is positive, the matrix is normalized
    to a spectral radius of rho and the scale shows the
    normalization.  If the spectral radius is zero, the returned
    matrix will have entries of 0, 1, and -1, and the scale is set
    to zero.

    Args:
        num_genes: Number of genes/nodes.
        prob_conn: Probability of connection.
        rho: The desired spectral radius.

    Returns:
        Adjacency matrix and its scale.
    """
    vals = np.asarray([-1.0, 0.0, 1.0])
    signed_edge_dist = rv_discrete(
        values=(np.arange(3), [prob_conn / 2, 1 - prob_conn, prob_conn / 2])
    )
    signed_edges = vals[signed_edge_dist.rvs(size=(num_genes, num_genes))]
    original_rho = max(abs(np.linalg.eigvals(signed_edges)))
    if original_rho:
        return signed_edges / original_rho * rho, rho / original_rho
    return signed_edges, 0
