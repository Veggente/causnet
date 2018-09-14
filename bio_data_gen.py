#!/usr/bin/env python3
"""Generate biologically plausible RNA-seq.
"""
import sys
import numpy as np
from scipy.integrate import odeint
from scipy.stats import norm
import pandas as pd
import argparse


def main(argv):
    num_experiments = 10
    csv_design_file = 'design.csv'
    num_replicates = 3
    num_times = 6
    max_in_deg = 3
    rand_seed = None
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
        "--gamma",
        help="fraction of power of condition dependent noise",
        type=float, default=0
        )
    args = parser.parse_args(argv)
    # The regulation coefficients have variance one regardless
    # of the margin.  So the SNR is [CITATION NEEDED]
    # num_times*max_in_degree/36/sigma**2.
    sigma = np.sqrt(max_in_deg*num_times/36/args.snr)
    sigma_c = np.sqrt(args.gamma)*sigma
    sigma_b = np.sqrt(1-args.gamma)*sigma
    adj_mat_file = args.adjmat
    if args.create:
        # Generate a random adjacency matrix file.
        margin = 0.5
        adj_mat = gen_adj_mat(args.num_core_genes, max_in_deg, margin)
        np.savetxt(adj_mat_file, adj_mat)
    gen_planted_edge_data(
        args.num_genes, adj_mat_file, sigma_c, sigma_b,
        num_experiments, args.exp, args.design, num_replicates,
        num_times, rand_seed
        )
    return


def main_old(argv):
    num_times = int(argv[0])
    sample_times = np.linspace(0.0, 50.0, num_times)
    beta = [0.05, 0.05, 0.1, 0.1]
    threshold = 10.0
    mm_const = [1.0, 1.0, 1.0, 1.0]
    dd_const = [0.1, 0.1, 0.1, 0.1]
    noise_std_dev = 0.0
    random_seed = 0
    xinit = [0, 0.6, 0.1, 0]
    num_exp = int(argv[1])
    num_main_genes = 4
    # The third argument, if any, is the number of extra genes.
    if len(argv) > 2:
        num_extra_genes = int(argv[2])
    else:
        num_extra_genes = 0
    x = []
    np.random.seed(random_seed)
    for idx_exp in range(num_exp):
        eta_leaf = np.random.rand(2) * 0.2
        x.append(generate_data(
            sample_times, xinit, beta, eta_leaf, threshold,
            mm_const, dd_const, noise_std_dev, num_extra_genes
            ))
    filename = (
        'data/bio-data-gen-t'+str(num_times)+'-m'+str(num_exp)+'-e'
        +str(num_extra_genes)+'.csv'
        )
    gene_ids = [
        'AT' + str(idx+1) for idx in range(num_main_genes+num_extra_genes)
        ]
    write_csv(x, filename, gene_ids)


def ode_drift(x, t, beta, eta_leaf, threshold, mm_const, dd_const):
    """Drift of the ODE.

    x -- gene expression levels.
        x[0] -- FT.
        x[1] -- TFL1.
        x[2] -- LFY.
        x[3] -- AP1.
    t -- time.
    beta -- maximum expression levels.
    eta_leaf -- rates of FT production per unit leaf.
    threshold -- threshold for the v1 input function.
    mm_const -- the Michaelis constants.
    dd_const -- degradation/dilution rates.
    """
    return np.array([
        (v1(t, eta_leaf, threshold) + beta[0]*x[2]/(mm_const[0]+x[2])
         - dd_const[0]*x[0]),
        beta[1]*x[2]/(mm_const[1]+x[2]+x[3]) - dd_const[1]*x[1],
        beta[2]*(x[0]+x[3])/(mm_const[2]+x[0]+x[1]+x[3]) - dd_const[2]*x[2],
        beta[3]*(x[0]+x[2])/(mm_const[3]+x[0]+x[1]+x[2]) - dd_const[3]*x[3]
        ])


def v1(t, eta_leaf, threshold):
    """Two-piece linear input signal to FT.

    t -- time.
    eta_leaf -- vector of rates of FT production per unit leaf.
    threshold -- change time of FT production rate.
    """
    if t < threshold:
        return eta_leaf[0] * t
    else:
        return eta_leaf[0]*threshold + eta_leaf[1]*(t-threshold)


def generate_data(sample_times, xinit, beta, eta_leaf, threshold,
                  mm_const, dd_const, noise_std_dev, num_extra_genes):
    """Generate one set of data with noise.

    sample_times -- list of sample times.
    xinit -- initial state of x.
    beta -- maximum expression levels.
    eta_leaf -- rates of FT production per unit leaf.
    threshold -- threshold for the piecewise linear v1 input function.
    mm_const -- the Michaelis constants.
    dd_const -- the degradation/dilution constants.
    noise_std_dev -- noise standard deviation.
    """
    assert(is_sorted(sample_times))
    x = odeint(ode_drift, xinit, sample_times,
               args=(beta, eta_leaf, threshold, mm_const, dd_const))
    if num_extra_genes:
        extra_genes = np.ones((len(sample_times), num_extra_genes))
        x = np.concatenate((x, extra_genes), axis=1)
    noise_x = np.random.normal(size=x.shape) * noise_std_dev
    x_w_noise = np.maximum(x + noise_x, np.zeros(x.shape))
    return x_w_noise


def is_sorted(l):
    """Test if a list is sorted in ascending order."""
    if all([l[i] <= l[i+1] for i in range(len(l)-1)]):
        return True
    else:
        return False


def write_csv(x, filename, gene_ids):
    """Write gene expression levels to file.

    x -- gene expression level matrix.
        x is <num_experiments>-by-<num_times>-by-<num_genes>.
        x[0][1][2] is the gene expression level of Gene 3 in Time 2 in
            Experiment 1.
    filename -- filename for the output.
    gene_ids -- gene IDs.
    """
    x_array = np.array(x)
    num_exp, num_times, num_genes = x_array.shape
    assert(len(gene_ids) == num_genes)
    with open(filename, 'w') as f:
        # Write headers. We follow the LD_G1_T1_A format. We also use
        # three identical replicates for now.
        for idx_exp in range(num_exp):
            for idx_time in range(num_times):
                exp_id = ',PP'+str(idx_exp+1)+'_G1_T'+str(idx_time+1)+'_'
                f.write(exp_id+'A'+exp_id+'B'+exp_id+'C')
        f.write('\n')
        # Write expressions for each gene.
        for idx_gene, gene_id in enumerate(gene_ids):
            f.write(gene_id)
            for idx_exp in range(num_exp):
                for idx_time in range(num_times):
                    f.write((
                        ','+str(x_array[idx_exp, idx_time, idx_gene]))*3
                        )
            f.write('\n')


def gen_planted_lindep_data(
        num_genes, a2, a3, sigma, num_experiments, csv_exp_file,
        csv_design_file, num_replicates, num_times, sd_mfp,
        rand_seed
        ):
    """Generate planted linear dependence data.

    TODO: Change documentation: dependence is not linear.

    Let n be the number of genes and let X_i^{jk}(t) be the
    expression level of gene i in replicate k of experiment j at
    sample time t.  The expression levels for genes 2, 3, ..., n are
    all i.i.d. Unif[0, 1], while expression levels for gene 1 is given
    by
    X_1^{jk}(t+1) = Phi^j(a_2^j*X_2^{jk}(t) + a_3^j*X_3^{jk}(t)
                    + sigma^j*W^{jk}(t+1)),
    where Phi^j is the Gaussian CDF with mean (a_2^j+a_3^j)/2 and
    variance ((a_2^j)**2+(a_3^j)**2)/2+(sigma^j)**2, a_2^j, a_3^j
    and sigma^j are the perturbed parameters, and W^{jk}(t+1) is
    standard Gaussian distributed.  Then X_1^{jk}(t+1) is also
    distributed over [0, 1].  Hence we have planted two linear
    dependences (gene 2 to gene 1 and gene 3 to gene 1) in the n-node
    graph.  Note ^ denotes superscript and ** denotes exponentiation.

    Args:
        num_genes: Number of genes.  Should be at least 3.
        a2: Linear dependence strength from X2 to X1.
        a3: Linear dependence strength from X3 to X1.
        sigma: Noise level of X1.
        num_experiments: Number of experiments.
        csv_exp_file: Path to output expression file.
        csv_design_file: Path to output design file.
        num_replicates: Number of replicates.
        num_times: Number of sample times.
        sd_mfp: Standard deviation of the Gaussian multifactorial
            perturbation of a2, a3 and sigma.
        rand_seed: Seed for random number generation.  None for the
            default clock seed (see
            https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.RandomState.html#numpy.random.RandomState).

    Returns:
        Write an expression file (csv_exp_file) and a design file
            (csv_design_file) in CSV format.
    """
    np.random.seed(rand_seed)
    expressions = []
    for i in range(num_experiments):
        # For each experiment, generate multifactorial perturbation.
        var_mfp = sd_mfp**2
        a2_perturbed = np.random.normal(a2, var_mfp)
        a3_perturbed = np.random.normal(a3, var_mfp)
        sigma_perturbed = np.absolute(np.random.normal(sigma, var_mfp))
        mean_x_0_unnormalized = (a2_perturbed+a3_perturbed)/2
        sd_x_0_unnormalized = np.sqrt(
            (a2_perturbed**2+a3_perturbed**2)/12 + sigma_perturbed**2
            )
        # Generate uniform expressions for all the genes.
        x = np.random.rand(num_genes, num_times, num_replicates)
        # Plant the two edges by modifying expression levels of gene 1.
        noise = np.random.normal(
            scale=sigma_perturbed, size=(num_times-1, num_replicates)
            )
        x_0_unnormalized = (
            a2_perturbed*x[1, :-1]+a3_perturbed*x[2, :-1]+noise
            )
        x_0_normalized = norm.cdf(
            (x_0_unnormalized-mean_x_0_unnormalized)
            / sd_x_0_unnormalized
            )
        x[0, 1:] = x_0_normalized
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
    with open(csv_design_file, 'w') as f:
        idx_sample = 0
        for i in range(num_experiments):
            for j in range(num_times):
                for k in range(num_replicates):
                    # Write the sample ID, condition, and the sample
                    # time to each line.
                    f.write(
                        sample_ids[idx_sample]+','+str(i)+','
                        +str(j)+'\n'
                        )
                    idx_sample += 1
    return


def gen_planted_edge_data(
        num_genes, adj_mat_file, sigma_c, sigma_b, num_experiments,
        csv_exp_file, csv_design_file, num_replicates, num_times,
        rand_seed, true_time=True
        ):
    """Generate data from the planted-edge model.

    Linear dependence is given by the adjacency matrix.  With
    Gaussian noise added, the sum is mapped back to Unif[0, 1]
    by Gaussian approximation.

    Args:
        num_genes: Number of genes.  Should be at least as large
            as the adjacency matrix.
        adj_mat_file: Adjacency matrix file.  The (i, j)th
            element is the regulation strength coefficient of
            gene i over gene j.
        sigma_c: Condition-dependent noise level.
        sigma_b: Condition-independent noise level.
        num_experiments: Number of experiments.
        csv_exp_file: Path to output expression file.
        csv_design_file: Path to output design file.
        num_replicates: Number of replicates.
        num_times: Number of sample times.
        rand_seed: Seed for random number generation.  None for the
            default clock seed (see
            https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.RandomState.html#numpy.random.RandomState).
        true_time: Indicator of using different individual for
            each sample.

    Returns:
        Write an expression file (csv_exp_file) and a
        design file (csv_design_file, if given) in CSV format.
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
        # Generate the condition-dependent standard noise.
        noise_c = np.random.randn(num_genes, num_times)
        if true_time:
            x = np.empty((num_genes, 0, num_replicates))
            for t in range(1, num_times+1):
                # Generate new independent trajectory up to time t.
                x_rep = gen_traj(num_replicates, t, adj_mat,
                                 sigma_b, sigma_c, noise_c)
                x_sample = x_rep[:, -1, :]
                x_sample_3d = x_sample[:, np.newaxis, :]
                x = np.concatenate((x, x_sample_3d), axis=1)
        else:
            x = gen_traj(num_replicates, num_times, adj_mat,
                         sigma_b, sigma_c, noise_c)
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


def phi_input(x_t_minus_1, adj_mat, sigma_b, sigma_c, noise_c_st):
    """Input function in Phi network model.

    Args:
        x_t_minus_1: An n-by-r array of expression levels at
            time t-1, where n is the number of genes and r is
            the number of replicates.
        adj_mat: An n-by-n array of the adjacency matrix.
        sigma_b: Condition-independent Noise level.
        sigma_c: Condition-dependent noise level.
        noise_c_st: Standard condition-dependent noise.
            An n-dim array.

    Returns:
        An n-by-r array of expression levels at time t.
    """
    num_genes, num_replicates = x_t_minus_1.shape
    # Influence of the regulating genes with mean subtracted.
    influence = (x_t_minus_1.T-0.5).dot(adj_mat)
    # AWGN with noise level sigma_b.
    noise = np.random.normal(
        scale=sigma_b, size=(num_replicates, num_genes)
        )
    # Standard deviations of the sum of influence and noise.
    sd_lin_expressions = np.sqrt(
        np.diag(adj_mat.T.dot(adj_mat))/12 + sigma_b**2
        + sigma_c**2
        )
    # Standardization of the linear expressions is done via
    # broadcasting.
    noise_c = sigma_c*noise_c_st.reshape(1, num_genes)
    standardized_lin_expressions = (
        (influence+noise+noise_c) / sd_lin_expressions
        )
    # Map the linear expressions back to [0, 1] by the CDF of
    # standard Gaussian (a.k.a. the Phi function).
    x_t = norm.cdf(standardized_lin_expressions).T
    return x_t


def gen_traj(num_replicates, num_times, adj_mat, sigma_b,
             sigma_c, noise_c):
    """Generate a new independent trajectory from time 0 up to time T.

    The same condition-dependent noise is shared across the replicates.

    Args:
        num_replicates: Number of replicates.
        num_times: Number of sample times.
        adj_mat: An n-by-n array of the adjacency matrix.
        sigma_b: Condition-independent noise level.
        sigma_c: Condition-dependent noise level.
        noise_c: Standard condition-dependent noise.  An n-by-T array.

    Returns:
        An n-by-T-by-r array of the expression level trajectory
        for n genes and r replicates.
        """
    num_genes = adj_mat.shape[0]
    # Generate constant 1/2 expression levels for all genes at time 0.
    x = np.ones((num_genes, 1, num_replicates))/2
    for t in range(1, num_times+1):
        x_new = phi_input(x[:, t-1, :], adj_mat, sigma_b, sigma_c,
                          noise_c[:, t-1])
        x = np.concatenate((x, x_new[:, np.newaxis, :]), axis=1)
    return x[:, 1:, :]


def gen_adj_mat(num_genes, max_in_deg, margin):
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

    Returns:
        A 2-d array of the adjacency matrix of the generated
        network.
    """
    adj_mat = np.zeros((num_genes, num_genes))
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
