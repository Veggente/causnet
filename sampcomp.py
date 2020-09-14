"""Calculate sample complexity of network reconstruction"""
from typing import Tuple, Optional, Union, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_discrete

plt.style.use("ggplot")


class NetworkHypothesisTesting:  # pylint: disable=too-many-instance-attributes
    """Network hypothesis testing."""

    def __init__(self):
        """Initialization.

        TODO: Set variables.
        """
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

    def sim_er_genie_bhatta_lb(  # pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
        self,
        num_genes: int,
        prob_conn: float,
        spec_rad: float,
        num_sims: int,
        num_cond: int,
        bayes: bool = True,
        stationary: bool = False,
        and_upper: bool = False,
    ) -> Union[
        Tuple[float, float],
        Tuple[Tuple[float, float], Tuple[float, float]],
        Tuple[
            Tuple[float, float],
            Tuple[float, float],
            Tuple[float, float],
            Tuple[float, float],
        ],
    ]:
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
            stationary: Use stationary initial condition.
            and_upper: Also gives the "upper bounds" based on genie's aid.

        Returns:
            Bhattacharyya lower bound, or lower and upper bounds.
        """
        lb_list = []
        if and_upper:
            ub_list = []
        if num_genes > 1:
            lb_cross_list = []
            ub_cross_list = []
        if bayes:
            prior = (1 - prob_conn, prob_conn)
        else:
            prior = (1 / 2, 1 / 2)
        for _ in range(num_sims):
            er_graph, weight = erdos_renyi(num_genes, prob_conn, spec_rad)
            # Autoregulation.  Genie tells everything except the self-edge (0, 0).
            auto_adj_mat = self.genie_hypotheses(er_graph, (0, 0), weight, spec_rad)
            if stationary:
                initial_auto_0, _ = asymptotic_cov_mat(
                    np.identity(num_genes),
                    auto_adj_mat[0],
                    self.sigma_en_sq + self.sigma_in_sq,
                    20,
                )
            else:
                initial_auto_0 = None
            auto_cov_mat_0 = gen_cov_mat(
                auto_adj_mat[0],
                self.sigma_in_sq,
                self.sigma_en_sq,
                self.samp_times,
                self.num_rep,
                self.one_shot,
                self.sigma_te_sq,
                initial=initial_auto_0,
            )
            if stationary:
                initial_auto_1, _ = asymptotic_cov_mat(
                    np.identity(num_genes),
                    auto_adj_mat[1],
                    self.sigma_en_sq + self.sigma_in_sq,
                    20,
                )
            else:
                initial_auto_1 = None
            auto_cov_mat_1 = gen_cov_mat(
                auto_adj_mat[1],
                self.sigma_in_sq,
                self.sigma_en_sq,
                self.samp_times,
                self.num_rep,
                self.one_shot,
                self.sigma_te_sq,
                initial=initial_auto_1,
            )
            rho_auto = bhatta_coeff(auto_cov_mat_0, auto_cov_mat_1)
            lb_list.append(
                self.lower_bound_on_error_prob(rho_auto, num_cond, prior=prior)
            )
            if and_upper:
                ub_list.append(self.upper_bound(rho_auto, num_cond))
            if num_genes > 1:
                # Cross regulation.
                cross_adj_mat = self.genie_hypotheses(
                    er_graph, (0, 1), weight, spec_rad
                )
                if stationary:
                    initial_cross_0, _ = asymptotic_cov_mat(
                        np.identity(num_genes),
                        cross_adj_mat[0],
                        self.sigma_en_sq + self.sigma_in_sq,
                        20,
                    )
                else:
                    initial_cross_0 = None
                cross_cov_mat_0 = gen_cov_mat(
                    cross_adj_mat[0],
                    self.sigma_in_sq,
                    self.sigma_en_sq,
                    self.samp_times,
                    self.num_rep,
                    self.one_shot,
                    self.sigma_te_sq,
                    initial=initial_cross_0,
                )
                if stationary:
                    initial_cross_1, _ = asymptotic_cov_mat(
                        np.identity(num_genes),
                        cross_adj_mat[1],
                        self.sigma_en_sq + self.sigma_in_sq,
                        20,
                    )
                else:
                    initial_cross_1 = None
                cross_cov_mat_1 = gen_cov_mat(
                    cross_adj_mat[1],
                    self.sigma_in_sq,
                    self.sigma_en_sq,
                    self.samp_times,
                    self.num_rep,
                    self.one_shot,
                    self.sigma_te_sq,
                    initial=initial_cross_1,
                )
                rho_cross = bhatta_coeff(cross_cov_mat_0, cross_cov_mat_1)
                lb_cross_list.append(
                    self.lower_bound_on_error_prob(rho_cross, num_cond, prior=prior)
                )
                if and_upper:
                    ub_cross_list.append(self.upper_bound(rho_cross, num_cond))
        auto_lb_stat = np.mean(lb_list), np.std(lb_list)
        if and_upper:
            auto_ub_stat = np.mean(ub_list), np.std(ub_list)
        if num_genes == 1:
            if and_upper:
                return auto_lb_stat, auto_ub_stat
            return auto_lb_stat
        cross_lb_stat = np.mean(lb_cross_list), np.std(lb_cross_list)
        if and_upper:
            cross_ub_stat = np.mean(ub_cross_list), np.std(ub_cross_list)
            return auto_lb_stat, cross_lb_stat, auto_ub_stat, cross_ub_stat
        return auto_lb_stat, cross_lb_stat

    def sim_er_bhatta(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        num_genes: int,
        prob_conn: float,
        spec_rad: Optional[float],
        num_sims: int,
        stationary: bool = False,
        step_size: float = 1,
        memory: bool = False,
        **kwargs,
    ) -> Tuple[float, float]:
        """Simulate average Bhattacharyya coefficient for ER graphs.

        Args:
            num_genes: Number of genes/nodes.
            prob_conn: Probability of connection.
            spec_rad: The desired spectral radius.  If None, do not
                rescale the network.
            num_sims: Number of simulations.
            stationary: Use stationary initial condition.
            step_size: Step size of the discrete-time system.
            memory: Add an identity matrix to the network matrix.
            **skip: int
                Number of times to skip for subsampling.

        Returns:
            Average Bhattacharyya coefficient and its standard deviation.
        """
        bhatta_list = []
        for _ in range(num_sims):
            er_graph, weight = erdos_renyi(num_genes, prob_conn, spec_rad)
            er_graph = er_graph * step_size
            if memory:
                er_graph = er_graph + np.identity(num_genes)
            cross_adj_mat = self.genie_hypotheses(er_graph, (0, 1), weight, spec_rad)
            if stationary:
                initial_0, _ = asymptotic_cov_mat(
                    np.identity(num_genes),
                    cross_adj_mat[0],
                    (self.sigma_en_sq + self.sigma_in_sq) * step_size,
                    20,
                )
            else:
                initial_0 = None
            cross_cov_mat_0 = gen_cov_mat(
                cross_adj_mat[0],
                self.sigma_in_sq * step_size,
                self.sigma_en_sq * step_size,
                self.samp_times,
                self.num_rep,
                self.one_shot,
                self.sigma_te_sq,
                initial=initial_0,
                **kwargs,
            )
            if stationary:
                initial_1, _ = asymptotic_cov_mat(
                    np.identity(num_genes),
                    cross_adj_mat[1],
                    (self.sigma_en_sq + self.sigma_in_sq) * step_size,
                    20,
                )
            else:
                initial_1 = None
            cross_cov_mat_1 = gen_cov_mat(
                cross_adj_mat[1],
                self.sigma_in_sq * step_size,
                self.sigma_en_sq * step_size,
                self.samp_times,
                self.num_rep,
                self.one_shot,
                self.sigma_te_sq,
                initial=initial_1,
                **kwargs,
            )
            rho_cross = bhatta_coeff(cross_cov_mat_0, cross_cov_mat_1)
            bhatta_list.append(rho_cross)
        bhatta_stat = np.mean(bhatta_list), np.std(bhatta_list)
        return bhatta_stat

    def genie_hypotheses(  # pylint: disable=no-self-use
        self,
        graph: np.ndarray,
        pos: Tuple[int, int],
        weight: float,
        spec_rad: Optional[float],
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
            elif spec_rad:
                scale = spec_rad
            else:
                scale = 1
            rademacher = np.random.binomial(1, 0.5) * 2 - 1
            adj_mat_1[pos] = scale * rademacher
        return adj_mat_0, adj_mat_1

    @staticmethod
    def lower_bound_on_error_prob(
        rho: float, num_cond: int, prior: Tuple[float, float] = (0.5, 0.5)
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

    @staticmethod
    def upper_bound(rho: float, num_cond: int) -> float:
        """Upper bound on half of sum of errors.

        Args:
            rho: Bhattacharyya coefficient.
            num_cond: Number of conditions.

        Returns:
            Upper bound.
        """
        return rho ** num_cond / 2


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
    """Calculates the small covariance matrix.

    If initial is None, we start with a zero matrix at time 0 and
    calculates the covariance matrix at time pair (tidx1, tidx2)
    (i.e., the (tidx1 + 1)st and the (tidx2 + 1)st time points).  If
    initial is not None, we start with initial as the covariance
    matrix at time 1 and calculates the covariance matrix at time pair
    (tidx1, tidx2).

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
        cov_mat = (
            np.linalg.matrix_power(adj_mat.T, tidx1)
            .dot(initial)
            .dot(np.linalg.matrix_power(adj_mat, tidx2))
        )
        times = (tidx1, tidx2)
    else:
        cov_mat = np.zeros(adj_mat.shape)
        times = (tidx1 + 1, tidx2 + 1)
    if (tidx1, ridx1) == (tidx2, ridx2):
        cov_mat += (sigma_in_sq + sigma_en_sq) * geom_sum_mat(
            adj_mat, *times
        ) + sigma_te_sq * np.identity(num_genes)
    elif ridx1 == ridx2 and not one_shot:
        cov_mat += (sigma_in_sq + sigma_en_sq) * geom_sum_mat(adj_mat, *times)
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


def gen_cov_mat(  # pylint: disable=too-many-arguments, too-many-locals
    adj_mat: np.ndarray,
    sigma_in_sq: float,
    sigma_en_sq: float,
    num_time: int,
    num_rep: int,
    one_shot: bool,
    sigma_te_sq: float,
    skip: int = 0,
    initial: Optional[np.ndarray] = None,
) -> Union[np.ndarray, List[np.ndarray]]:
    """Generates covariance matrix.

    Generates covariance matrix for the observations of possibly
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

    Returns:
        The covariance matrix.
    """
    if initial is not None and (num_rep != 1 or one_shot):
        raise ValueError(
            "Can only take initial covariance matrix for single replicate multi-shot sampling."
        )
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


def gen_cov_mat_w_skips(
    adj_mat: np.ndarray,
    num_tran: int,
    driv_var: float,
    obs_var: float,
    skips: List[int],
) -> List[np.ndarray]:
    """Generates covariance matrices for different sampling rates.

    Generates covariance matrices for the observations of a
    discrete-time linear time-invariant system with Gaussian driving
    and observation noises with different sampling rates.

    Args:
        adj_mat: Network adjacency matrix.
        num_tran: T, the number of transitions at base sampling rate.
            The observations are at times 0, 1, 2, ..., T.
        driv_var: Driving noise variance.
        obs_var: Observation noise variance.
        skips: Number of time slots skipped in subsampling.

    Returns:
        The covariance matrices.
    """
    num_genes = adj_mat.shape[0]
    initial = asymptotic_cov_mat(np.identity(num_genes), adj_mat, driv_var, 20)[0]
    cov_mats = []
    for this_skip in skips:
        num_time = int(num_tran / (this_skip + 1)) + 1
        cov_mats.append(
            gen_cov_mat(
                adj_mat,
                driv_var,
                0,
                num_time,
                1,
                False,
                obs_var,
                skip=this_skip,
                initial=initial,
            )
        )
    return cov_mats


def erdos_renyi(
    num_genes: int, prob_conn: float, spec_rad: Optional[float] = 0.8
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
    if original_spec_rad and spec_rad:
        return signed_edges / original_spec_rad * spec_rad, spec_rad / original_spec_rad
    return signed_edges, original_spec_rad


def asymptotic_cov_mat(
    initial: np.ndarray, adj_mat: np.ndarray, sigma_sq: float, num_iter: int
) -> Tuple[np.ndarray, float]:
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
        new_cov_mat = adj_mat.T.dot(last_cov_mat).dot(adj_mat) + sigma_sq * np.identity(
            adj_mat.shape[0]
        )
        if i == num_iter - 1:
            difference = np.linalg.norm(new_cov_mat - last_cov_mat)
        last_cov_mat = new_cov_mat
    return last_cov_mat, difference


def bhatta_w_small_step(
    step_size: float, total_time: float, skip: int, obs_var: float
) -> float:
    """Calculates Bhattacharyya coefficient with small step size.

    Samples are at times [eta, 2 * eta, 3 * eta, ..., int(T / eta) *
    eta], where eta and T are the step size and the total time
    interval.  A BHT of two 2x2 matrices are used.

    Args:
        step_size: Step size.
        total_time: Total time interval.
        skip: Number of skipped samples per sample.
        obs_var: Observation noise variance level.

    Returns:
        Bhattacharyya coefficient.
    """
    network_ht = NetworkHypothesisTesting()
    network_ht.hypotheses = [
        np.array([[-1, 1], [0, -1]]),
        np.array([[-1, -1], [0, -1]]),
    ]
    num_genes = 2
    projector_mat = [
        np.identity(num_genes) + step_size * hypo for hypo in network_ht.hypotheses
    ]
    stationary = [
        asymptotic_cov_mat(np.identity(num_genes), this_mat, step_size, 20)[0]
        for this_mat in projector_mat
    ]
    cov_mat = [
        gen_cov_mat(
            this_mat,
            0,
            step_size,
            int(total_time / step_size / (skip + 1)),
            1,
            False,
            0,
            skip=skip,
            initial=stationary[idx],
        )
        for idx, this_mat in enumerate(projector_mat)
    ]
    if obs_var:
        cov_mat = [
            this_mat + obs_var / step_size * np.identity(this_mat.shape[0])
            for this_mat in cov_mat
        ]
    return bhatta_coeff(*cov_mat)
