"""Test oCSE algorithm."""
import unittest
import numpy as np
import causnet_bslr
import script_causnet


class TestOCSE(unittest.TestCase):
    """Test oCSE algorithm in causnet_bslr module."""

    def test_ocse_two_genes(self):
        """Test oCSE with a two-gene network."""
        adj_mat = np.array([[0.9, -0.5], [0, 0.9]])
        num_times = 1000
        num_genes = 2
        num_perm = 1000
        # TODO: Use script_causnet.gen_lin_gaussian().
        data_cell = [np.empty((num_times, 2))]
        driving_noise = np.random.randn(num_times, 2)
        data_cell[0][0, :] = driving_noise[0, :]
        for i in range(1, num_times):
            data_cell[0][i, :] = data_cell[0][i-1, :].dot(adj_mat)+driving_noise[i, :]
        parents, signs = causnet_bslr.ocse(data_cell, num_perm)
        full_network = np.zeros((num_genes, num_genes))
        for j in range(num_genes):
            for idx, i in enumerate(parents[j]):
                full_network[i, j] = signs[j][idx]
        np.testing.assert_array_equal(full_network, np.sign(adj_mat))

    def test_get_errors(self):
        """Test error calculator."""
        decision = np.array([[1, 0, -1], [0, 0, 1]])
        truth = np.array([[0, 0, 1], [-1, 0, 0]])
        error_rates = np.array([1/2, 2/4])
        np.testing.assert_array_equal(error_rates, script_causnet.get_errors(decision, truth))
