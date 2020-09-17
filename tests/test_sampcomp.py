"""Test sampcomp."""
import unittest
import math
import numpy as np
import sampcomp  # pylint: disable=import-error


class TestSampComp(unittest.TestCase):
    """Test sampcomp module"""

    def test_bhatta(self):
        """Test sampcomp.bhatta_bounds."""
        network_ht = sampcomp.NetworkHypothesisTesting()
        float_bounds = network_ht.bhatta_bound()
        int_bounds = (
            math.floor(float_bounds[0]),
            math.ceil(float_bounds[1]),
        )
        self.assertEqual(int_bounds, (1503, 4170))

    @staticmethod
    def test_gen_cov_mat():
        """Test sampcomp.gen_cov_mat."""
        network_ht = sampcomp.NetworkHypothesisTesting()
        cov_mat = sampcomp.gen_cov_mat(
            network_ht.hypotheses[0],
            network_ht.sigma_in_sq,
            network_ht.sigma_en_sq,
            network_ht.samp_times,
            network_ht.num_rep,
            network_ht.one_shot,
            network_ht.sigma_te_sq,
        )
        np.testing.assert_array_equal(
            cov_mat,
            np.array([[3, 0, 0, 0.1], [0, 3, 0, 0], [0, 0, 3, 0], [0.1, 0, 0, 3.02]]),
        )

    def test_bhatta_coeff(self):
        """Test sampcomp.bhatta_coeff."""
        correct_rho = np.sqrt(81.45 / 81.54)
        network_ht = sampcomp.NetworkHypothesisTesting()
        cov_mat_0 = sampcomp.gen_cov_mat(
            network_ht.hypotheses[0],
            network_ht.sigma_in_sq,
            network_ht.sigma_en_sq,
            network_ht.samp_times,
            network_ht.num_rep,
            network_ht.one_shot,
            network_ht.sigma_te_sq,
        )
        cov_mat_1 = sampcomp.gen_cov_mat(
            network_ht.hypotheses[1],
            network_ht.sigma_in_sq,
            network_ht.sigma_en_sq,
            network_ht.samp_times,
            network_ht.num_rep,
            network_ht.one_shot,
            network_ht.sigma_te_sq,
        )
        self.assertAlmostEqual(correct_rho, sampcomp.bhatta_coeff(cov_mat_0, cov_mat_1))

    @staticmethod
    def test_erdos_renyi():
        """Test sampcomp.erdos_renyi."""
        np.random.seed(0)
        np.testing.assert_almost_equal(
            sampcomp.erdos_renyi(5, 0.2, 0.5)[0],
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.5, 0.0],
                    [0.0, 0.0, 0.0, 0.5, -0.5],
                    [-0.5, -0.5, 0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        )

    def test_er_sampcomp(self):
        """Test ER graph sample complexity lower bound."""
        np.random.seed(0)
        num_genes = 1
        sigma_en_sq = 1
        sigma_in_sq = 0
        sigma_te_sq = 0
        samp_times = 2
        num_rep = 1
        prob_conn = 0.5
        spec_rad = 0.5
        num_sims = 2
        num_cond = 3
        # Correct Bhattacharyya coefficient per condition.
        rho = 4 / np.sqrt(17)
        network_ht = sampcomp.NetworkHypothesisTesting()
        network_ht.sigma_en_sq = sigma_en_sq
        network_ht.sigma_in_sq = sigma_in_sq
        network_ht.sigma_te_sq = sigma_te_sq
        network_ht.samp_times = samp_times
        network_ht.num_rep = num_rep
        sim_lower_bound = network_ht.sim_er_genie_bhatta_lb(
            num_genes, prob_conn, spec_rad, num_sims, num_cond
        )
        correct_lower_bound = network_ht.lower_bound_on_error_prob(rho, num_cond)
        self.assertAlmostEqual(correct_lower_bound, sim_lower_bound[0])

    def test_low_spec_rad(self):
        """Test ER simulations with low spectral radius."""
        sim_net = sampcomp.NetworkHypothesisTesting()
        sim_net.sigma_in_sq = 0
        sim_net.sigma_te_sq = 0
        num_genes = 2
        prob_conn = 0.05
        spec_rad = 0.1
        samp_time = 2
        num_cond = 2
        num_sims = 100
        sim_net.samp_times = samp_time
        avg_err, _ = sim_net.sim_er_genie_bhatta_lb(
            num_genes, prob_conn, spec_rad, num_sims, num_cond
        )[1]
        self.assertTrue(avg_err <= prob_conn * 1.1)

    @staticmethod
    def test_sub_samp():
        """Test subsampling."""
        adj_mat = np.array([[0, 1], [0, 0]])
        cov_mat = sampcomp.gen_cov_mat(adj_mat, 0, 1, 2, 1, False, 0, 1)
        cov_mat_target = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 2]]
        )
        np.testing.assert_array_equal(cov_mat, cov_mat_target)

    @staticmethod
    def test_small_cov():
        """Tests small covariance matrix generation."""
        adj_mat = np.array([[0, 1], [0, 0]])
        cov_mat = sampcomp.cov_mat_small(adj_mat, 0, 1, 0, 0, 0, 0, 0, False)
        np.testing.assert_array_equal(cov_mat, np.identity(2))

    @staticmethod
    def test_initial_dist():
        """Test covariance matrix generation with initial distribution."""
        network_ht = sampcomp.NetworkHypothesisTesting()
        cov_mat = sampcomp.gen_cov_mat(
            network_ht.hypotheses[0],
            network_ht.sigma_in_sq,
            network_ht.sigma_en_sq,
            network_ht.samp_times,
            network_ht.num_rep,
            False,
            network_ht.sigma_te_sq,
            initial=np.array([[1, 0.5], [0.5, 1]]),
        )
        np.testing.assert_array_equal(
            cov_mat,
            np.array(
                [
                    [2, 0.5, 0, 0.1],
                    [0.5, 2, 0, 0.05],
                    [0, 0, 3, 0],
                    [0.1, 0.05, 0, 3.01],
                ]
            ),
        )

    def test_stationary_cov_mat(self):
        """Tests calculating stationary distribution iteratively."""
        network_ht = sampcomp.NetworkHypothesisTesting()
        initial = np.identity(2)
        stat_cov_mat, difference = sampcomp.asymptotic_cov_mat(
            initial, network_ht.hypotheses[0], 2, 10
        )
        np.testing.assert_array_equal(stat_cov_mat, np.array([[2, 0], [0, 2.02]]))
        self.assertEqual(difference, 0)

    @staticmethod
    def test_cov_w_stationary_skip():
        """Tests covariance matrix generation with stationary initial condition and skips."""
        network_ht = sampcomp.NetworkHypothesisTesting()
        cov_mats = sampcomp.gen_cov_mat_w_skips(
            network_ht.hypotheses[0], 2, 2, 0, [0, 1]
        )
        correct_cov_mats = [
            np.array(
                [
                    [2, 0, 0, 0.2, 0, 0],
                    [0, 2.02, 0, 0, 0, 0],
                    [0, 0, 2, 0, 0, 0.2],
                    [0.2, 0, 0, 2.02, 0, 0],
                    [0, 0, 0, 0, 2, 0],
                    [0, 0, 0.2, 0, 0, 2.02],
                ]
            ),
            np.diag([2, 2.02, 2, 2.02]),
        ]
        np.testing.assert_array_equal(cov_mats[0], correct_cov_mats[0])
        np.testing.assert_array_equal(cov_mats[1], correct_cov_mats[1])

    def test_bhatta_skip(self):
        """Tests Bhattacharyya coefficient with skipped data."""
        network_ht = sampcomp.NetworkHypothesisTesting()
        step_size = 0.02
        num_genes = 16
        spec_rad = 0.8
        er_graph, weight = sampcomp.erdos_renyi(num_genes, 1 / 4, spec_rad)
        er_graph = er_graph * step_size
        er_graph += np.identity(num_genes)
        er_graph_pair = network_ht.genie_hypotheses(er_graph, (0, 1), weight, spec_rad)
        cov_mats = [
            sampcomp.gen_cov_mat_w_skips(er_graph_pair[i], 2, 2, 0, [0, 1])
            for i in range(2)
        ]
        rho = sampcomp.bhatta_coeff(cov_mats[0][0], cov_mats[1][0])
        rho_skip = sampcomp.bhatta_coeff(cov_mats[0][1], cov_mats[1][1])
        self.assertTrue(rho <= rho_skip)

    def test_bhatta_stat(self):
        """Tests BC with stationary initial distribution."""
        rho = sampcomp.bhatta_w_small_step(0.5, 1, 0, 0, 0)
        cov1 = np.array(
            [
                [2 / 3, 2 / 9, 1 / 3, 4 / 9],
                [2 / 9, 28 / 27, 1 / 9, 17 / 27],
                [1 / 3, 1 / 9, 2 / 3, 2 / 9],
                [4 / 9, 17 / 27, 2 / 9, 28 / 27],
            ]
        )
        cov2 = np.array(
            [
                [2 / 3, -2 / 9, 1 / 3, -4 / 9],
                [-2 / 9, 28 / 27, -1 / 9, 17 / 27],
                [1 / 3, -1 / 9, 2 / 3, -2 / 9],
                [-4 / 9, 17 / 27, -2 / 9, 28 / 27],
            ]
        )
        self.assertAlmostEqual(rho, sampcomp.bhatta_coeff(cov1, cov2))
