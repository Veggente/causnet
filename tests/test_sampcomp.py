"""Test sampcomp"""
import unittest
import sampcomp
import math
import numpy as np


class TestSampComp(unittest.TestCase):
    def test_bhatta(self):
        network_ht = sampcomp.NetworkHypothesisTesting()
        float_bounds = network_ht.bhatta_bound()
        int_bounds = (
            math.floor(float_bounds[0]),
            math.ceil(float_bounds[1]),
        )
        self.assertEqual(int_bounds, (1503, 4170))

    def test_gen_cov_mat(self):
        network_ht = sampcomp.NetworkHypothesisTesting()
        cov_mat = network_ht.gen_cov_mat(
            network_ht.hypotheses[0],
            network_ht.sigma_in_sq,
            network_ht.sigma_en_sq,
            network_ht.samp_times,
            network_ht.num_rep,
            network_ht.one_shot,
            network_ht.sigma_te_sq,
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                cov_mat,
                np.array(
                    [
                        [3, 0, 0, 0.1],
                        [0, 3, 0, 0],
                        [0, 0, 3, 0],
                        [0.1, 0, 0, 3.02],
                    ]
                ),
            )
        )

    def test_bhatta_coeff(self):
        correct_rho = np.sqrt(81.45 / 81.54)
        network_ht = sampcomp.NetworkHypothesisTesting()
        cov_mat_0 = network_ht.gen_cov_mat(
            network_ht.hypotheses[0],
            network_ht.sigma_in_sq,
            network_ht.sigma_en_sq,
            network_ht.samp_times,
            network_ht.num_rep,
            network_ht.one_shot,
            network_ht.sigma_te_sq,
        )
        cov_mat_1 = network_ht.gen_cov_mat(
            network_ht.hypotheses[1],
            network_ht.sigma_in_sq,
            network_ht.sigma_en_sq,
            network_ht.samp_times,
            network_ht.num_rep,
            network_ht.one_shot,
            network_ht.sigma_te_sq,
        )
        self.assertAlmostEqual(
            correct_rho, sampcomp.bhatta_coeff(cov_mat_0, cov_mat_1)
        )
