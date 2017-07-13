#!/usr/bin/env python3
"""CaSPIAN with brute force."""
import numpy as np
import scipy.stats
import itertools


def testing():
    data_cell_single = np.array(np.mat('3 4 5; 6 7 8'))
    data_cell = [
        np.array([[0.836334,  0.08015632,  0.001396,  0.64343897],
                  [0.63934881,  0.44191633,  0.70205871,  0.47928562],
                  [0.01185488,  0.69447034,  0.49794843,  0.73974141]]).T,
        np.array([[0.179556,  0.07568321,  0.66753137],
                  [0.62753963,  0.22615587,  0.03522489],
                  [0.39816049,  0.42056573,  0.10281385]])
        ]
    num_time_lags = 2
    num_gene = 3
    matrix_2 = get_shifted_matrix(data_cell, num_time_lags)
    for j in range(num_gene):
        for i in range(num_time_lags + 1):
            print('matrix_2[:, {0}, {1}] ='.format(j, i),
                  matrix_2[:, j, i])
    data_normalized = normalize(matrix_2)
    for j in range(num_gene):
        for i in range(num_time_lags + 1):
            print('data_normalized[:, {0}, {1}] ='.format(j, i),
                  data_normalized[:, j, i])
    phi = data_normalized
    potential_parents = compressive_sensing(phi)
    print(potential_parents)
    np.random.seed(0)
    print("Testing standardize():")
    data_cell_2 = [
        np.random.rand(3, 4), np.random.rand(3, 4), np.random.rand(3, 4)
        ]
    print("Before standardization:\n", data_cell_2)
    data_cell_2_st = standardize(data_cell_2)
    print("After standardization:\n", data_cell_2_st)


def caspian(data_cell, num_time_lags, max_in_degree, significance_level,
            self_reg=False, st=False):
    # Preprocess the data.
    # Standardization.
    if st:
        data_cell_processed = standardize(data_cell)
    else:
        data_cell_processed = data_cell
    shifted_data = get_shifted_matrix(data_cell_processed, num_time_lags)
    phi = normalize(shifted_data)
    # Find candidate links with brute force using numpy.linalg.inv.
    # The data of latest time point is the target.
    # The rest are used for prediction; i.e., the Phi matrix.
    # Compressive sensing with errors.
    potential_parents, errors = compressive_sensing(phi, self_reg,
                                                    max_in_degree)
    # Use Granger causality to remove any insignificant links.  Then return
    # the resulting graph with signs, and the p-values if the significance
    # level is set to be zero.
    return granger(phi, potential_parents, errors,
                   significance_level, self_reg)


def get_shifted_matrix(data_cell, num_time_lags, is_single_mat=False):
    """Return a 3-dimensional matrix for compressive sensing.

    Axis 0 -- distinct virtual experiments.
    Axis 1 -- genes.
    Axis 2 -- time lags.
    """
    # Generate a single block row for a single-matrix input.
    if is_single_mat:
        num_time_points, num_genes = data_cell.shape
        sliding_window_height = num_time_points - num_time_lags
        # Generate the uninitiated shifted matrix.
        shifted_matrix = np.empty([
            sliding_window_height, num_genes, num_time_lags + 1
            ])
        for pos in range(num_time_lags + 1):
            shifted_matrix[:, :, pos] = (
                data_cell[pos:pos+sliding_window_height, :]
                )
        return shifted_matrix
    else:  # data_cell is a list of multiple matrices.
        # Recursively run the single-matrix versions.
        num_genes = data_cell[0].shape[1]
        shifted_matrix = np.empty([0, num_genes, num_time_lags+1])
        for data_page in data_cell:
            new_block_row = get_shifted_matrix(
                data_page, num_time_lags, is_single_mat=True
                )
            shifted_matrix = np.concatenate((
                shifted_matrix, new_block_row
                ), axis=0)
        return shifted_matrix


def normalize(data):
    """Normalize the 3-dimensional data along first axis."""
    _, dim_1, dim_2 = data.shape
    data_normalized = np.empty(data.shape)
    for idx_1 in range(dim_1):
        for idx_2 in range(dim_2):
            data_vec_temp = data[:, idx_1, idx_2]
            data_vec_centered = data_vec_temp - np.mean(data_vec_temp)
            data_vec_norm = np.linalg.norm(data_vec_centered)
            # Normalize if the norm is not zero.
            if data_vec_norm:
                data_vec_normalized = data_vec_centered / data_vec_norm
            else:
                # Otherwise, keep the all-zero vectors unchanged.
                data_vec_normalized = data_vec_centered
            data_normalized[:, idx_1, idx_2] = data_vec_normalized
    return data_normalized


def compressive_sensing(phi, self_reg=False, max_in_degree=0):
    """Compressive sensing with brute force."""
    num_genes = phi.shape[1]
    parents = []
    errors = []
    # Return all other genes if max_in_degree is 0.
    if not max_in_degree:
        for idx_gene in range(num_genes):
            all_other_genes = (
                list(range(idx_gene))
                + list(range(idx_gene + 1, num_genes))
                )
            parents.append(all_other_genes)
            _, err = whole_gene_lsa(phi, idx_gene, all_other_genes,
                                    self_reg)
            # A single-element list is used in accordance with the
            # max_degree > 0 case.
            errors.append([err])
    else:
        # For each target gene, try all combinations of k = max_in_degree
        # parent genes. Pick the best combination.
        for idx_gene in range(num_genes):
            errors.append([])
            parents.append([])
            all_other_genes = [g for g in range(num_genes) if g != idx_gene]
            for combo_tuple in itertools.combinations(all_other_genes,
                                                      max_in_degree):
                combo = list(combo_tuple)
                _, err = whole_gene_lsa(phi, idx_gene, combo, self_reg)
                # Note that we distinguish the case of errors[idx_gene]
                # being False (the empty list, meaning that gene has not
                # been considered yet) and the case of errors[idx_gene]
                # being [0] (zero error).
                if errors[idx_gene]:
                    if errors[idx_gene][0] > err:
                        errors[idx_gene][0] = err
                        parents[idx_gene] = combo
                else:
                    errors[idx_gene].append(err)
                    parents[idx_gene] = combo
    return parents, errors


# TODO: Better interface.
def granger(phi, potential_parents, errors, significance_level=0.0,
            self_reg=False):
    # Setting the significance level to be the default 0.0 indicates
    # actual p-values to be returned instead of filtered parents, in which
    # case the returned p_values and signs are for all the
    # potential_parents.
    if significance_level:
        parents = []
    else:
        parents = potential_parents
        all_p_values = []
    signs = []
    num_experiments = phi.shape[0]
    num_time_lags = phi.shape[2] - 1
    for idx_gene, its_pot_parents in enumerate(potential_parents):
        if significance_level:
            parents.append([])
        else:
            all_p_values.append([])
        signs.append([])
        if self_reg:
            num_genes_unrestricted = len(its_pot_parents) + 1
        else:
            num_genes_unrestricted = len(its_pot_parents)
        dof = (
            num_time_lags, (
                num_experiments
                - num_genes_unrestricted * num_time_lags
                - 1
                )
            )
        assert(dof[0] > 0 and dof[1] > 0)
        for idx_parent in its_pot_parents:
            restricted_parents = [
                p for p in its_pot_parents if p != idx_parent
                ]
            _, restricted_error = whole_gene_lsa(
                phi, idx_gene, restricted_parents, self_reg
                )
            f_stat = (
                (restricted_error**2 - errors[idx_gene][0]**2)
                / dof[0]
                / (errors[idx_gene][0]**2)
                * dof[1]
                )
            p_value = 1 - scipy.stats.f.cdf(f_stat, dof[0], dof[1])
            if significance_level and p_value < significance_level:
                parents[idx_gene].append(idx_parent)
            if not significance_level:
                all_p_values[idx_gene].append(p_value)
        num_parents = len(parents[idx_gene])
        coeff, err = whole_gene_lsa(phi, idx_gene, parents[idx_gene],
                                    self_reg)
        coeff_2d = coeff.reshape(num_time_lags, num_parents)
        for idx_parent, parent in enumerate(parents[idx_gene]):
            if (coeff_2d[:, idx_parent] > 0).all():
                # Sign is positive (activation).
                signs[idx_gene].append(1)
            elif (coeff_2d[:, idx_parent] < 0).all():
                # Sign is negative (repression).
                signs[idx_gene].append(-1)
            else:
                # Sign is undetermined.
                signs[idx_gene].append(0)
    # TODO: Probably need to resolve the multiple-return-statement issue.
    if not significance_level:
        return parents, signs, all_p_values
    else:
        return parents, signs


def whole_gene_lsa(phi, this_gene, parent_genes, self_reg=False):
    size_phi = phi.shape
    num_experiments = size_phi[0]
    num_time_lags = size_phi[2] - 1
    regressand = phi[:, this_gene, num_time_lags]
    assert(this_gene not in parent_genes)
    # The previous time lags of the target gene are also used in regression.
    # Note the genes in the regressor are not ordered.
    if self_reg:
        regressor_3d = phi[:, np.array(parent_genes + [this_gene]), :-1]
        num_genes_to_fit_with = len(parent_genes) + 1
    else:
        regressor_3d = phi[:, parent_genes, :-1]
        num_genes_to_fit_with = len(parent_genes)
    regressor = regressor_3d.reshape(
        num_experiments, num_genes_to_fit_with * num_time_lags
        )
    coeff_all, residual = lsa(regressand, regressor)
    # Remove the coefficients for the target gene if self-regulation is on.
    if self_reg:
        assert(len(coeff_all)
               == num_genes_to_fit_with * num_time_lags)
        idx_keep = [x for x in range(len(coeff_all))
                    if (x+1) % num_genes_to_fit_with != 0]
        coeff = coeff_all[idx_keep]
    else:
        coeff = coeff_all
    return coeff, np.linalg.norm(residual)


def lsa(regressand, regressor):
    """Least-squares approximation to fit columns of regressor to those of
    regressand.
    """
    # Check the dimensions of regressand and regressor.
    assert(regressand.shape[0] == regressor.shape[0])
    if not regressor.size:
        coeff = np.empty((0, regressand.size//regressand.shape[0]))
        residual = regressand
    else:
        coeff = np.linalg.pinv(regressor).dot(regressand)
        residual = regressand-regressor.dot(coeff)
    return coeff, residual


def standardize(data_cell):
    """Time-dependent standardization.

    Note this is not applicable for varying time length data.
    """
    # Need at least two experiments for standardization.
    assert(len(data_cell) > 1)
    num_time_points, num_genes = data_cell[0].shape
    data_3d_array = np.empty([0, num_time_points, num_genes])
    for data_block in data_cell:
        assert((num_time_points, num_genes) == data_block.shape)
        data_3d_array = np.concatenate((
            data_3d_array, np.array([data_block])), axis=0
            )
    data_3d_array_st = normalize(data_3d_array)
    data_cell_st = []
    for idx in range(data_3d_array_st.shape[0]):
        data_cell_st.append(data_3d_array_st[idx, :, :])
    return data_cell_st


if __name__ == "__main__":
    testing()
