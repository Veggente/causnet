#!/usr/bin/env python3
"""Generate biologically plausible RNA-seq data.

Functions:
    main: Generate biologically plausible data.
    gen_planted_edge_data: Phi-network model.
    gen_adj_mat: Generate adjacency matrix.
"""
import sys
import numpy as np
from scipy.integrate import odeint
from scipy.stats import norm
import pandas as pd
import argparse


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("adjmat", help="adjacency matrix")
    parser.add_argument("exp", help="expression level file")
    parser.add_argument("-c", "--create",
                        help="create new adjacency matrix",
                        action="store_true")
    parser.add_argument("-d", "--design",
                        help="path to save design file to",
                        default="")
    parser.add_argument(
        "--num-core-genes",
        help="number of core genes (only with -c)", type=int,
        default=5
        )
    parser.add_argument(
        "--num-genes",
        help="total number of genes (only with -c)",
        type=int, default=20
        )
    parser.add_argument("-s", "--snr",
                        help="signal to noise ratio",
                        type=float, default=1)
    parser.add_argument(
        "--num-experiments", help="number of experiments",
        type=int, default=10
        )
    parser.add_argument(
        "--num-replicates",
        help="number of biological replicates",
        type=int, default=3
        )
    parser.add_argument(
        "--num-times", help="number of sample times",
        type=int, default=6
        )
    parser.add_argument(
        "--max-in-deg", help="maximum in-degree",
        type=int, default=3
        )
    parser.add_argument(
        "--margin", help="edge strength margin",
        type=float, default=0.5
        )
    parser.add_argument(
        "--rand-seed", help="random number generator seed",
        type=int, default=None
        )
    args = parser.parse_args()
    # The regulation coefficients have variance one regardless
    # of the margin.  So the SNR is [CITATION NEEDED]
    # num_times*max_in_degree/36/sigma**2.
    sigma = np.sqrt(
        args.max_in_deg * args.num_times / 36 / args.snr
        )
    adj_mat_file = args.adjmat
    if args.create:
        # Generate a random adjacency matrix file.
        adj_mat = gen_adj_mat(
            args.num_core_genes, args.max_in_deg, args.margin,
            args.rand_seed
            )
        np.savetxt(adj_mat_file, adj_mat)
    gen_planted_edge_data(
        args.num_genes, adj_mat_file, sigma, args.num_experiments,
        args.exp, args.design, args.num_replicates, args.num_times,
        args.rand_seed
        )
    return


def gen_planted_edge_data(
        num_genes, adj_mat_file, sigma, num_experiments, csv_exp_file,
        csv_design_file, num_replicates, num_times, rand_seed
        ):
    """Generate data from the planted-edge (Phi-network) model.

    TODO: Add multifactorial perturbation with and without time series.

    Dependence is given by the adjacency matrix.  With
    Gaussian noise added, the sum is mapped back to Unif[0, 1]
    by Gaussian approximation.

    Args:
        num_genes: Number of genes.  Should be at least as large
            as the adjacency matrix.
        adj_mat_file: Adjacency matrix file.  The (i, j)th
            element is the regulation strength coefficient of
            gene i over gene j.  Must be square.
        sigma: Noise level.
        num_experiments: Number of experiments.
        csv_exp_file: Path to output expression file.
        csv_design_file: Path to output design file.
        num_replicates: Number of replicates.
        num_times: Number of sample times.
        rand_seed: Seed for random number generation.  None for the
            default clock seed (see
            https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.RandomState.html#numpy.random.RandomState).

    Returns:
        Write an expression file (csv_exp_file, if given) and a
            design file (csv_design_file) in CSV format.
    """
    # Load adjacency matrix.
    adj_mat = np.loadtxt(adj_mat_file, delimiter=' ')
    # Check size of adjacency matrix.
    num_genes_in_adj_mat = adj_mat.shape[0]
    if num_genes < num_genes_in_adj_mat:
        print('The specified number of genes is smaller than the'
              'size of the adjacency matrix.')
        return 1
    if num_genes > num_genes_in_adj_mat:
        adj_mat_big = np.zeros((num_genes, num_genes))
        adj_mat_big[
            :num_genes_in_adj_mat, :num_genes_in_adj_mat
            ] = adj_mat
        adj_mat = adj_mat_big
    np.random.seed(rand_seed)
    expressions = []
    for i in range(num_experiments):
        # Generate i.i.d. uniform expressions for all the genes at
        # time 1.
        x = np.random.rand(num_genes, 1, num_replicates)
        for t in range(1, num_times):
            x_t_minus_1 = x[:, t-1, :]
            # Influence of the regulating genes with mean subtracted.
            influence = (x_t_minus_1.T-0.5).dot(adj_mat)
            # AWGN with noise level sigma.
            noise = np.random.normal(
                scale=sigma, size=(num_replicates, num_genes)
                )
            # Standard deviations of the sum of influence and noise.
            sd_lin_expressions = np.sqrt(
                np.diag(adj_mat.T.dot(adj_mat))/12 + sigma**2
                )
            # Standardization of the linear expressions is done via
            # broadcasting.
            standardized_lin_expressions = (
                (influence+noise) / sd_lin_expressions
                )
            # Map the linear expressions back to [0, 1] by the CDF of
            # standard Gaussian (a.k.a. the Phi function).
            x_t = norm.cdf(standardized_lin_expressions).T
            x = np.concatenate((x, x_t[:, np.newaxis, :]), axis=1)
        expressions.append(x)
    # Output expression file.
    sample_ids = ['Sample'+str(i) for i in
                  range(num_replicates*num_experiments*num_times)]
    genes = ['Gene'+str(i) for i in range(num_genes)]
    flattened_exp = np.empty((num_genes, 0))
    for i in range(num_experiments):
        flattened_exp = np.concatenate((
            flattened_exp, expressions[i].reshape(
                num_genes, num_times*num_replicates
                )
            ), axis=1)
    df = pd.DataFrame(data=flattened_exp, columns=sample_ids,
                      index=genes)
    df.to_csv(csv_exp_file)
    # Output design file.
    if csv_design_file:
        with open(csv_design_file, 'w') as f:
            idx_sample = 0
            for i in range(num_experiments):
                for j in range(num_times):
                    for k in range(num_replicates):
                        # Write the sample ID, condition, and the
                        # sample time to each line.
                        f.write(
                            sample_ids[idx_sample]+','+str(i)+','
                            +str(j)+'\n'
                            )
                        idx_sample += 1
    return 0


def gen_adj_mat(num_genes, max_in_deg, margin, rand_seed):
    """Generate adjacency matrix.

    Assume the in-degree is uniformly distributed over 0, 1, 2,
    ..., max_in_deg.  Regulation strength coefficients are
    Gaussian shifted away from the origin by the margin with
    variance one.

    Args:
        num_genes: The number of genes.
        max_in_deg: The maximum in-degree.
        margin: The margin of regulation strength coefficients
            from zero.
            margin must be between 0 and 1.
            The standard deviation of the Gaussian distribution
            before the shift is then determined by the margin so
            that the actual variance stays one.
        rand_seed: RNG seed.

    Returns:
        A 2-d array of the adjacency matrix of the generated
        network.
    """
    adj_mat = np.zeros((num_genes, num_genes))
    np.random.seed(rand_seed)
    in_degrees = np.random.randint(max_in_deg+1, size=num_genes)
    # Standard deviation of the unshifted Gaussians.
    sd = np.sqrt(1-(1-2/np.pi)*margin**2)-np.sqrt(2/np.pi)*margin
    for i in range(num_genes):
        other_genes = [x for x in range(num_genes) if x != i]
        regulators = np.random.choice(
            other_genes, size=in_degrees[i], replace=False
            )
        st_gaussians = np.random.randn(in_degrees[i])
        coeffs = sd*st_gaussians + margin*np.sign(st_gaussians)
        adj_mat[regulators, i] = coeffs
    return adj_mat


if __name__ == "__main__":
    main(sys.argv[1:])
