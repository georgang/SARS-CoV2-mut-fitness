import os
import numpy as np
import pandas as pd
from Bio import SeqIO
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from helper import load_mut_counts, tolerant_orfs
from general_linear_models.fit_general_linear_models import add_predictions

plt.rcParams.update({'font.size': 12})

# Colors of mutation types in the zoom-in plots
mut_type_cols = {'AC': 'blue', 'AG': 'orange', 'AT': 'green', 'CA': 'red', 'CG': 'purple', 'CT': 'brown',
                 'GA': 'pink', 'GC': 'gray', 'GT': 'cyan', 'TA': 'magenta', 'TC': 'lime', 'TG': 'teal'}
mut_type_cols_2 = {'noncoding': 'purple', 'synonymous': 'blue', 'nonsynonymous': 'orange'}

# Load secondary structure information, TODO: The illustration in the experimental paper is for Vero cells
df_sec_str = pd.read_csv(
    f'/Users/georgangehrn/Desktop/SARS-CoV2-mut-fitness/sec_stru_data/data/sec_structure_Huh7.txt',
    header=None, sep='\s+').drop(columns=[2, 3, 5])
df_sec_str.rename(columns={0: 'nt_site', 1: 'nt_type', 4: 'unpaired'}, inplace=True)

# obtained from get_gene_boundaries.py
gene_boundaries = [(266, 13480, 'ORF1a'),
                   (13468, 21552, 'ORF1b'),
                   (21563, 25381, 'S'),
                   (25393, 26217, 'ORF3a'),
                   (26245, 26469, 'E'),
                   (26523, 27188, 'M'),
                   (27202, 27384, 'ORF6'),
                   (27394, 27756, 'ORF7a'),
                   (27756, 27884, 'ORF7b'),
                   (27894, 28256, 'ORF8'),
                   (28274, 29530, 'N'),
                   (29558, 29671, 'ORF10')]


def get_conserved_regions(only_known=False):
    TRS_motif = "ACGAAC"  # consensus sequence of transcription regulatory sites
    margin = 0
    ref = SeqIO.read('../human_data/reference.gb', 'genbank')

    motifs = []
    pos = 1
    while pos > 0:
        pos = ref.seq.find(TRS_motif, pos)
        if pos > 1:
            motifs.append(pos)
            pos += 1

    conserved_regions = {}
    for mi, mpos in enumerate(motifs):
        conserved_regions[f'TRS {mi + 1}'] = (mpos - margin, mpos + len(TRS_motif) + margin)

    # attenuator hairpin
    conserved_regions['hairpin'] = (ref.seq.find("ATGCTTCA"), ref.seq.find("CGTTTTT"))
    # attenuator hairpin
    conserved_regions['slippery seq'] = (ref.seq.find("TTTAAACG"), ref.seq.find("TTTAAACG") + 7)
    # 3 stem pseudoknot
    conserved_regions['3-stem-pseudoknot'] = (ref.seq.find("GCGGTGT"), ref.seq.find("TTTTGA", 13474))
    if not only_known:
        # center of E
        conserved_regions['E-center'] = (26330, 26360)
        # end of M
        conserved_regions['M-end'] = (27170, 27200)

    conserved_vector = np.zeros(len(ref.seq), dtype=bool)
    for r in conserved_regions.values():
        conserved_vector[r[0]:r[1]] = True

    return conserved_vector, conserved_regions


def log_p_of_f_k(sites, f_hat, f_vals, a_left, b_left, a_right, b_right, sigma_n, sigma_s, sigma_f):
    """
    :param sites: nucleotide sites k for which log(p(f_k)) is calculated, dimension (n_sites,)
    :param f_vals: fitness values for which log(p(f_k)) is calculated, dimension (n_f,)
    :return: log(p(f_k)) for all sites and fitness values, dimension (n_sites, n_f)
    """

    sigma_f = sigma_f[sites - 1].reshape(-1, 1)

    var_n = sigma_n ** 2
    var_s = sigma_s ** 2

    inv_var_n = 1 / var_n
    inv_var_s = 1 / var_s

    # Get left and right messages, all of dimension (n_sites,)
    a_left = a_left[sites - 1].reshape(-1, 1)
    b_left = b_left[sites - 1].reshape(-1, 1)
    a_right = a_right[sites - 1].reshape(-1, 1)
    b_right = b_right[sites - 1].reshape(-1, 1)

    # Reshape f_vals and f_hat
    f_vals = f_vals.reshape(1, -1)
    f_hat = f_hat.reshape(-1, 1)

    # Put everything together
    msg_left = (b_left + f_vals * inv_var_n) ** 2 / (4 * a_left) - 0.5 * f_vals ** 2 * (inv_var_s + inv_var_n)
    msg_right = ((b_right + f_vals * inv_var_n) ** 2 / (4 * a_right)) - 0.5 * f_vals ** 2 * (
                inv_var_n - 1 / (var_n + var_s))
    p_n_given_f_k = - (f_hat - f_vals) ** 2 / (2 * sigma_f ** 2)

    return msg_left + msg_right + p_n_given_f_k


def get_most_likely_f_value(f_values, log_p_values):
    """
    :param f_values: dimension (n_f)
    :param log_p_values: dimension (n_sites, n_f)
    :return x_max: dimension (n_sites,)
    """

    # Vandermonde matrix for quadratic polynomial fitting
    V = np.vander(f_values, 3)

    # Solve least squares problem to find polynomial coefficients for each site
    coefficients = np.linalg.lstsq(V, log_p_values.T, rcond=None)[0]

    # Extract coefficients a and b, which are needed to find x_max
    a = coefficients[0]
    b = coefficients[1]
    x_max = -b / (2 * a)

    return x_max


def translate_position(excluded_sites, original_position):
    """
    :param excluded_sites: list of sites that have excluded mutations and which are neither at the start or the end
    :param original_position: nt position in the original reference sequence
    :return: position in the genome that has all excluded sites removed
    """
    position = np.searchsorted(excluded_sites, original_position)
    new_site = original_position - 265 - position
    return new_site


def get_left_msgs(f_hat, sigma_n, sigma_s, sigma_f):
    var_n = sigma_n ** 2
    var_s = sigma_s ** 2
    var_f = sigma_f ** 2

    inv_var_n = 1 / var_n
    inv_var_s = 1 / var_s
    inv_var_f = 1 / var_f

    a = np.zeros(len(f_hat))
    b = np.zeros(len(f_hat))

    a[0] = 0.5 * (inv_var_n + inv_var_s + inv_var_f[0] - 1 / (var_n + var_s))
    b[0] = f_hat[0] * inv_var_f[0]

    for i in range(1, len(f_hat)):
        a[i] = 0.5 * (2 * inv_var_n + inv_var_f[i] + inv_var_s - 1 / (var_n + var_s) - 1 / (2 * a[i - 1] * var_n ** 2))
        b[i] = b[i - 1] / (2 * a[i - 1] * var_n) + f_hat[i] * inv_var_f[i]

    return a, b


def get_right_msgs(f_hat, sigma_n, sigma_s, sigma_f):
    var_n = sigma_n ** 2
    var_s = sigma_s ** 2
    var_f = sigma_f ** 2

    inv_var_n = 1 / var_n
    inv_var_s = 1 / var_s
    inv_var_f = 1 / var_f

    a = np.zeros(len(f_hat))
    b = np.zeros(len(f_hat))

    a[-1] = 0.5 * (inv_var_n + inv_var_s + inv_var_f[-1])
    b[-1] = f_hat[-1] * inv_var_f[-1]

    for i in range(2, len(f_hat) + 1):
        a[-i] = 0.5 * (
                    2 * inv_var_n + inv_var_f[-i] + inv_var_s - 1 / (var_n + var_s) - 1 / (2 * a[-i + 1] * var_n ** 2))
        b[-i] = b[-i + 1] / (2 * a[-i + 1] * var_n) + f_hat[-i] * inv_var_f[-i]

    return a, b


def get_probabilistic_fitness_estimates_deprecated(clade, hyperparams):
    # Load all mutation counts and add predictions
    df = load_mut_counts(clade=clade, mut_types='all')
    df = add_predictions(df.copy(), clade=clade)

    # Remove excluded sites at the beginning and the end of the genome (new length N = 29674 - 265 = 29409)
    df = df[(df['nt_site'] > 265) & (df['nt_site'] < 29675)]

    # Remove and remember all other sites that have any excluded mutations
    sites_with_exclusions = df.groupby('nt_site')['exclude'].any()
    sites_with_exclusions = sites_with_exclusions[sites_with_exclusions].index.values
    df = df[~df['nt_site'].isin(sites_with_exclusions)]

    # Get average fitness effect (f_hat, cf. notes) at all sites that have no excluded mutations, dimension (N,)
    # - version 1 (averaging fitness estimates at site)
    f_hat = np.log(df['actual_count'].values + 0.5) - df['pred_log_count'].values
    f_hat = (f_hat[0:-2:3] + f_hat[1:-1:3] + f_hat[2::3]) / 3
    # - version 2 (fitness of summed counts)
    actual_counts_averaged = df.groupby('nt_site')['actual_count'].mean().values
    df['pred_count'] = np.exp(df['pred_log_count']) - 0.5
    predicted_counts_averaged = df.groupby('nt_site')['pred_count'].mean().values
    f_hat = np.log(actual_counts_averaged + 0.5) - np.log(predicted_counts_averaged + 0.5)

    # Set variances
    expected_counts = np.exp(df['pred_log_count'].values)
    if isinstance(hyperparams['sigma_f'], (int, float)):
        sigma_f = np.full((len(f_hat),), hyperparams['sigma_f'])
    elif hyperparams['sigma_f'] == 'test':
        sigma_f = np.full((len(f_hat),), 0.0001)  # 1.5 * expected_counts / (expected_counts + 0.5)
        coding_mask = ~(df['four_fold_degenerate'].values[::3] + df['noncoding'].values[::3])
        sigma_f[coding_mask] = 10000
    sigma_n = hyperparams['sigma_n']
    sigma_s = hyperparams['sigma_s']

    # Compute coefficients from iterative calculation of right/left messages, each of dimension (N,)
    a_left, b_left = get_left_msgs(f_hat, sigma_n, sigma_s, sigma_f)
    a_right, b_right = get_right_msgs(f_hat, sigma_n, sigma_s, sigma_f)

    # original and translated (new) positions of non-excluded sites
    original_positions = np.arange(266, 29674 + 1)
    mask = ~np.isin(original_positions, sites_with_exclusions)
    original_positions = original_positions[mask]
    new_positions = translate_position(sites_with_exclusions, original_positions)

    # Get original, non-probabilistic fitness estimates
    original_fitness_estimates = f_hat[new_positions - 1]

    # Define fitness values to sweep over (x-axis of optimization task)
    f_values = np.arange(-7, 5, 0.2)

    # Calculate quantity proportional to the log probability
    # new_positions is of dimension (n_sites,), f_values is of dimension (n_f,)
    log_p_values = log_p_of_f_k(new_positions, original_fitness_estimates, f_values, a_left, b_left, a_right, b_right,
                                sigma_n, sigma_s, sigma_f)

    # Get most likely fitness value at every site in the window
    # f_values is of dimension (n_f,), log_p_values is of dimension (n_sites, n_f)
    new_fitness_estimates = get_most_likely_f_value(f_values, log_p_values)

    # Only return the original positions which are noncoding/four-fold degenerate
    return original_positions, new_fitness_estimates, original_positions[~coding_mask], original_fitness_estimates[
        ~coding_mask]


def solve_tridiagonal_lse(d, e, f, y):
    """
    Inverts a tridiagonal matrix A given its diagonal, superdiagonal, and subdiagonal elements.
    Parameters:
    d (array): diagonal elements of the matrix
    e (array): superdiagonal elements of the matrix
    f (array): subdiagonal elements of the matrix
    y (array): right-hand side of LSE
    Returns:
    x (array): solution to the system Ax = y
    """

    # Create the tridiagonal matrix in sparse format
    A = sp.diags([f, d, e], offsets=[-1, 0, 1], format='csc')

    # Solve the system Ax = y
    x = spla.spsolve(A, y)

    return x


def solve_lse(sigma, rho, tau, g):
    # Precompute 1/tau**2
    inv_tau_sq = 1 / tau ** 2

    # Prepare tridiagonal coefficient matrix
    diagonal = 1 / sigma ** 2 + 1 / rho ** 2 + np.append(0, inv_tau_sq[:-1]) + np.append(inv_tau_sq[:-1], 0)
    superdiagonal = - inv_tau_sq[:-1]
    subdiagonal = - inv_tau_sq[:-1]

    # Prepare y in the linear system of equations Af = y
    y = g / rho ** 2

    # Solve linear system of equations to get probabilistic fitness estimates
    f = solve_tridiagonal_lse(diagonal, superdiagonal, subdiagonal, y)

    return f


def get_probabilistic_fitness_estimates(clade, hyperparameters):
    """
    Parameters:
        (...)
    Returns:
        nucleotide_positions (array): nucleotide positions >265 and <29675
        f_i (array): probabilistic fitness estimates
        (...)
    """

    # Load all mutation counts and add predicted counts
    df = load_mut_counts(clade=clade, mut_types='all')
    df = add_predictions(df.copy(), clade=clade)

    # Remove excluded sites at the beginning and the end of the genome (new length = 29674 - 265 = 29409)
    df = df[(df['nt_site'] > 265) & (df['nt_site'] < 29675)]
    L = 29409

    # Mask for noncoding mutations
    mask_noncoding = df['noncoding'].values
    # Mask for synonymous mutations
    mask_synonymous = df['synonymous'].values
    # Mask for mutations that are in N;ORF9b and synonymous in N
    mask_orf9b = ((df['gene'] == 'N;ORF9b') & (
            df['clade_founder_aa'].apply(lambda x: x[0]) == df['mutant_aa'].apply(lambda x: x[0]))).values
    # Mask for sites which are in stop codon tolerant open reading frames
    pattern = '|'.join(tolerant_orfs)
    mask_tolerant = (df['gene'].str.contains(pattern)).values
    # Mask for non-excluded sites
    mask_nonexcluded = ~df['exclude'].values
    # Combine masks
    df['mask'] = (mask_noncoding + mask_synonymous + mask_orf9b + mask_tolerant) * mask_nonexcluded
    # Mask for sites with at least one included mutation
    mask_rho = df.groupby('nt_site')['mask'].any()

    def custom_agg(group):
        if group['mask'].any():
            actual_counts_sum = group.loc[group['mask'], 'actual_count'].sum()
            predicted_counts_sum = (np.exp(group.loc[group['mask'], 'pred_log_count']) - 0.5).sum()
            return np.log(actual_counts_sum + 0.5) - np.log(predicted_counts_sum + 0.5)
        else:
            actual_counts_sum = group['actual_count'].sum()
            predicted_counts_sum = (np.exp(group['pred_log_count']) - 0.5).sum()
            return np.log(actual_counts_sum + 0.5) - np.log(predicted_counts_sum + 0.5)

    # Get observed fitness effect at every site
    g_i = df.groupby('nt_site').apply(custom_agg).values

    # nucleotide positions
    nucleotide_positions = np.arange(266, 29674 + 1)

    # Prepare list of fitness estimates with different hyperparameters
    f_i = []

    for hyperparams in hyperparameters:
        # Set sigma_i (cf. notes)
        sigma_i = np.full((L,), hyperparams['sigma_s'])

        # Set rho_i (cf. notes)
        rho_i = np.full((L,), hyperparams['sigma_f'][0])
        rho_i[~mask_rho] = hyperparams['sigma_f'][1]

        # Set tau_i (cf. notes)
        tau_i = np.full((L,), hyperparams['sigma_n'])

        # Get probabilistic fitness estimates
        f_i.append(solve_lse(sigma_i, rho_i, tau_i, g_i))

    # Return nucleotide sites >265 and <29675, the corresponding probabilistic fitness estimates, and the naive
    # fitness estimate at all sites which have at least one included mutation
    return nucleotide_positions, f_i, nucleotide_positions[mask_rho], g_i[mask_rho]


def plot_fitness_estimates_deprecated(df, df_nonexc, probabilistic_pos, probabilistic_estims, original_pos, original_estims,
                             hyperparams, selected_trace, first_site, last_site, min_size=60, min_fit=-7, max_fit=5):
    """
    :param probabilistic_pos: nt positions for which there is a probabilistic fitness estimate (29'247 non-excluded sites)
    :param probabilistic_estims: f_k which maximizes the posterior probability p(f_k|{n_i})
    :param original_pos: nt positions where all three mutations are noncoding/synonymous
    :param original_estims: ln(sum of actual counts at this site + 0.5) - ln(sum of predicted counts at this site + 0.5)
    :param hyperparams: variances used in the model
    :param first_site: first nucleotide position in the plot
    :param last_site: last nucleotide position in the plot
    :return:
    """

    if last_site - first_site > 250:
        print("region too large for plot")
    else:
        # Enlarge region if too small
        if last_site - first_site < min_size:
            diff = min_size - last_site + first_site
            first_site = first_site - int(diff / 2)
            last_site = last_site + int(diff / 2)

        # Get average fitness estimates and standard deviations across all sites present in the selected trace
        mean_new, stdev_new = np.average(probabilistic_estims[selected_trace]), np.std(
            probabilistic_estims[selected_trace])
        mean_old, stdev_old = np.average(original_estims), np.std(original_estims)

        # Define figure size
        w = 0.125 * (last_site - first_site)
        h = 5
        plt.figure(figsize=(w, h), dpi=300)

        # Add mean and standard deviations of selected trace
        plt.axhline(0, linestyle='-', color='gray', lw=2, alpha=0.5)
        plt.axhline(mean_new, color='green', linestyle='-', alpha=0.3 + selected_trace * 0.3)
        plt.axhline(mean_new + OUTLIER_COND * stdev_new, color='green', linestyle='--',
                    alpha=0.3 + selected_trace * 0.3)
        plt.axhline(mean_new - OUTLIER_COND * stdev_new, color='green', linestyle='--',
                    alpha=0.3 + selected_trace * 0.3)
        plt.axhline(mean_old, color='red', linestyle='-', alpha=0.5)
        plt.axhline(mean_old + OUTLIER_COND * stdev_old, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        plt.axhline(mean_old - OUTLIER_COND * stdev_old, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

        # Plot all fitness estimates
        for i in range(len(hyperparams)):
            plt.plot(probabilistic_pos, probabilistic_estims[i], linestyle='-', marker='o',
                     label=f"$\sigma_{{n}}=${hyperparams[i]['sigma_n']}, $\sigma_{{f}}=${hyperparams[i]['sigma_f'][0]}",
                     markersize=3,
                     linewidth=2, color='green', alpha=0.3 + i * 0.3)
        plt.plot(original_pos, original_estims, linestyle='-', marker='o', label="naive estimate",
                 markersize=3, linewidth=1.5, color='red', alpha=0.5)

        # Set limits of plot
        plt.xlim((first_site, last_site))
        plt.ylim((min_fit, max_fit))

        # Add labels and title
        plt.xlabel('nucleotide position')
        plt.ylabel('fitness effect')

        # Add gene boundaries to plot, TODO: Make all of this a reusable function
        colors = cm.get_cmap('tab20', len(gene_boundaries))
        for i, (start, end, name) in enumerate(gene_boundaries):
            plt.barh(y=min_fit + 1.1 + (i % 2) * 0.3, width=end - start, left=start, height=0.3, color=colors(i),
                     alpha=0.7)
            if first_site <= start <= last_site:
                plt.text(start, min_fit + 1.1 + (i % 2) * 0.3, name, ha='left', va='center', fontsize=10)
            elif first_site <= end <= last_site or (start < first_site and end > last_site):
                plt.text(first_site, min_fit + 1.1 + (i % 2) * 0.3, name, ha='left', va='center', fontsize=10)

        # Add known conserved regions
        _, conserved_regions = get_conserved_regions()
        conserved_regions = [(value[0], value[1], key) for key, value in conserved_regions.items()]
        for i, (start, end, name) in enumerate(conserved_regions):
            if ((start <= first_site <= end) or (start <= last_site <= end)) or (
                    (first_site <= start <= last_site) and (first_site <= end <= last_site)):
                plt.barh(y=min_fit + 1.7 + (i % 3) * 0.3, width=end - start, left=start, height=0.3,
                         color='orange', alpha=0.6)
                plt.text(max(start, first_site), min_fit + 1.7 + (i % 3) * 0.3, name, ha='left',
                         va='center', fontsize=10)

        # Get secondary structure information to write on the plot
        df_help = df_sec_str[(df_sec_str['nt_site'] >= first_site) & (df_sec_str['nt_site'] <= last_site)]
        sites_help = df_help['nt_site'].values
        unpaired = (df_help['unpaired'].values == 0)
        paired = 1 - unpaired

        # Read in 21J reference sequence, TODO: Find a better solution that not only works for 21J
        with open(f'21J_refseq', 'r') as file:
            ref_seq = file.read()

        # Get corresponding section of the reference sequence
        nt_seq = ref_seq[int(first_site - 1):int(last_site)]

        # Add parent nucleotides to plot
        sites_help = np.arange(first_site, last_site + 1)
        for i, letter in enumerate(nt_seq):
            plt.text(sites_help[i], min_fit + 0.2, letter, ha='center', va='bottom', fontsize=11)

        # Add pairing information to plot
        plt.bar(sites_help, unpaired * 0.3, bottom=min_fit + 0.6, color='red', alpha=0.7)
        plt.bar(sites_help, paired * 0.3, bottom=min_fit + 0.6, color='blue', alpha=0.7)

        # Add every 10th nucleotide position as x-tick
        sites = probabilistic_pos[(probabilistic_pos >= first_site) & (probabilistic_pos <= last_site)]
        xticks = range(min(sites) + 10 - min(sites) % 10, max(sites), 10)
        plt.xticks(xticks)

        # Add vertical gridlines
        plt.minorticks_on()
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
        plt.grid(which='major', axis='x', linestyle='-', linewidth=0.5, color='black')
        plt.grid(which='minor', axis='x', linestyle=':', linewidth=0.5, color='gray')

        # Add legend and show plot
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Define figure size
        w = 0.125 * (last_site - first_site)
        h = 5
        plt.figure(figsize=(w, h), dpi=300)

        # Extract data for all mutations that fall into this region
        df = df[(df['nt_site'] >= first_site) & (df['nt_site'] <= last_site)]

        # Extract nt positions and un-/corrected fitness estimates of all mutations that fall into this region
        nt_positions = df['nt_site'].values
        fitness_estimates = {'corr': np.log(df['actual_count'].values + 0.5) - df['pred_log_count'].values,
                             'uncorr': np.log(df['actual_count'].values + 0.5) - np.log(
                                 df['expected_count'].values + 0.5)}

        # Color datapoints according to mutation type
        mut_type = df['nt_mutation'].apply(lambda x: x[0] + x[-1])
        scatter_colors = mut_type.apply(lambda x: mut_type_cols[x])

        # Add scatter plot
        plt.scatter(nt_positions, fitness_estimates['corr'], c=scatter_colors, s=50, edgecolors='black',
                    alpha=0.8)

        # Add mean and standard deviations of selected trace
        plt.axhline(0, linestyle='-', color='gray', lw=2, alpha=0.5)
        plt.axhline(mean_new, color='green', linestyle='-', alpha=0.3 + selected_trace * 0.3)
        plt.axhline(mean_new + OUTLIER_COND * stdev_new, color='green', linestyle='--',
                    alpha=0.3 + selected_trace * 0.3)
        plt.axhline(mean_new - OUTLIER_COND * stdev_new, color='green', linestyle='--',
                    alpha=0.3 + selected_trace * 0.3)
        plt.axhline(mean_old, color='red', linestyle='-', alpha=0.5)
        plt.axhline(mean_old + OUTLIER_COND * stdev_old, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        plt.axhline(mean_old - OUTLIER_COND * stdev_old, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

        # Plot all fitness estimates
        i = selected_trace
        plt.plot(probabilistic_pos, probabilistic_estims[i], linestyle='-', marker='o',
                 label=f"$\sigma_{{n}}=${hyperparams[i]['sigma_n']}, $\sigma_{{f}}=${hyperparams[i]['sigma_f'][0]}",
                 markersize=3,
                 linewidth=2, color='green', alpha=0.3 + i * 0.3)
        plt.plot(original_pos, original_estims, linestyle='-', marker='', label="naive estimate",
                 markersize=3, linewidth=1.5, color='red', alpha=0.5)

        # Set limits of plot
        plt.xlim((first_site, last_site))
        plt.ylim((min_fit, max_fit))

        # Add labels and title
        plt.xlabel('nucleotide position')
        plt.ylabel('fitness effect')

        # Add gene boundaries to plot, TODO: Make all of this a reusable function
        colors = cm.get_cmap('tab20', len(gene_boundaries))
        for i, (start, end, name) in enumerate(gene_boundaries):
            plt.barh(y=min_fit + 1.1 + (i % 2) * 0.3, width=end - start, left=start, height=0.3, color=colors(i),
                     alpha=0.7)
            if first_site <= start <= last_site:
                plt.text(start, min_fit + 1.1 + (i % 2) * 0.3, name, ha='left', va='center', fontsize=10)
            elif first_site <= end <= last_site or (start < first_site and end > last_site):
                plt.text(first_site, min_fit + 1.1 + (i % 2) * 0.3, name, ha='left', va='center', fontsize=10)

        # Add known conserved regions
        _, conserved_regions = get_conserved_regions()
        conserved_regions = [(value[0], value[1], key) for key, value in conserved_regions.items()]
        for i, (start, end, name) in enumerate(conserved_regions):
            if ((start <= first_site <= end) or (start <= last_site <= end)) or (
                    (first_site <= start <= last_site) and (first_site <= end <= last_site)):
                plt.barh(y=min_fit + 1.7 + (i % 3) * 0.3, width=end - start, left=start, height=0.3,
                         color='orange', alpha=0.6)
                plt.text(max(start, first_site), min_fit + 1.7 + (i % 3) * 0.3, name, ha='left',
                         va='center', fontsize=10)

        # Get secondary structure information to write on the plot
        df_help = df_sec_str[(df_sec_str['nt_site'] >= first_site) & (df_sec_str['nt_site'] <= last_site)]
        sites_help = df_help['nt_site'].values
        unpaired = (df_help['unpaired'].values == 0)
        paired = 1 - unpaired

        # Read in 21J reference sequence, TODO: Find a better solution that not only works for 21J
        with open(f'21J_refseq', 'r') as file:
            ref_seq = file.read()

        # Get corresponding section of the reference sequence
        nt_seq = ref_seq[int(first_site - 1):int(last_site)]

        # Add parent nucleotides to plot
        sites_help = np.arange(first_site, last_site + 1)
        for i, letter in enumerate(nt_seq):
            plt.text(sites_help[i], min_fit + 0.2, letter, ha='center', va='bottom', fontsize=11)

        # Add pairing information to plot
        plt.bar(sites_help, unpaired * 0.3, bottom=min_fit + 0.6, color='red', alpha=0.7)
        plt.bar(sites_help, paired * 0.3, bottom=min_fit + 0.6, color='blue', alpha=0.7)

        # Cross every dot in the scatter plot which has zero counts
        actual_counts = df['actual_count'].values
        zero_count_sites = nt_positions[actual_counts == 0]
        zero_count_fitness = fitness_estimates['corr'][actual_counts == 0]
        plt.scatter(zero_count_sites, zero_count_fitness, color='black', marker='x', alpha=0.5, label='0 counts')

        # Add every 10th nucleotide position as x-tick
        sites = probabilistic_pos[(probabilistic_pos >= first_site) & (probabilistic_pos <= last_site)]
        xticks = range(min(sites) + 10 - min(sites) % 10, max(sites), 10)
        plt.xticks(xticks)

        # Add legend for mutation type colors
        legend_handles = [mpatches.Patch(color=color, label=mut_type) for mut_type, color in mut_type_cols.items()]
        plt.legend(handles=legend_handles, fontsize='xx-small', loc='lower right')

        # Add vertical gridlines
        plt.minorticks_on()
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
        plt.grid(which='major', axis='x', linestyle='-', linewidth=0.5, color='black')
        plt.grid(which='minor', axis='x', linestyle=':', linewidth=0.5, color='gray')

        # Add legend and show plot
        plt.tight_layout()
        plt.show()

        # Define figure size
        w = 0.125 * (last_site - first_site)
        h = 5
        plt.figure(figsize=(w, h), dpi=300)

        # Extract data for all mutations that fall into this region
        df = df_nonexc[(df_nonexc['nt_site'] >= first_site) & (df_nonexc['nt_site'] <= last_site)]

        # Extract nt positions and un-/corrected fitness estimates of all mutations that fall into this region
        nt_positions = df['nt_site'].values
        fitness_estimates = {'corr': np.log(df['actual_count'].values + 0.5) - df['pred_log_count'].values,
                             'uncorr': np.log(df['actual_count'].values + 0.5) - np.log(
                                 df['expected_count'].values + 0.5)}

        # Color datapoints according to mutation type
        colors = np.full(len(df), "", dtype='<U20')
        colors[df['noncoding'].values == True] = "noncoding"
        colors[df['synonymous'].values == True] = "synonymous"
        colors[colors == ''] = 'nonsynonymous'
        scatter_colors = np.vectorize(mut_type_cols_2.get)(colors)

        # Add scatter plot
        plt.scatter(nt_positions, fitness_estimates['corr'], c=scatter_colors, s=50, edgecolors='black',
                    alpha=0.8)

        # Add mean and standard deviations of selected trace
        plt.axhline(0, linestyle='-', color='gray', lw=2, alpha=0.5)
        plt.axhline(mean_new, color='green', linestyle='-', alpha=0.3 + selected_trace * 0.3)
        plt.axhline(mean_new + OUTLIER_COND * stdev_new, color='green', linestyle='--',
                    alpha=0.3 + selected_trace * 0.3)
        plt.axhline(mean_new - OUTLIER_COND * stdev_new, color='green', linestyle='--',
                    alpha=0.3 + selected_trace * 0.3)
        plt.axhline(mean_old, color='red', linestyle='-', alpha=0.5)
        plt.axhline(mean_old + OUTLIER_COND * stdev_old, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        plt.axhline(mean_old - OUTLIER_COND * stdev_old, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

        # Plot all fitness estimates
        i = selected_trace
        plt.plot(probabilistic_pos, probabilistic_estims[i], linestyle='-', marker='o',
                 label=f"$\sigma_{{n}}=${hyperparams[i]['sigma_n']}, $\sigma_{{f}}=${hyperparams[i]['sigma_f'][0]}",
                 markersize=3,
                 linewidth=2, color='green', alpha=0.3 + i * 0.3)
        plt.plot(original_pos, original_estims, linestyle='-', marker='', label="naive estimate",
                 markersize=3, linewidth=1.5, color='red', alpha=0.5)

        # Set limits of plot
        plt.xlim((first_site, last_site))
        plt.ylim((min_fit, max_fit))

        # Add labels and title
        plt.xlabel('nucleotide position')
        plt.ylabel('fitness effect')

        # Add gene boundaries to plot, TODO: Make all of this a reusable function
        colors = cm.get_cmap('tab20', len(gene_boundaries))
        for i, (start, end, name) in enumerate(gene_boundaries):
            plt.barh(y=min_fit + 1.1 + (i % 2) * 0.3, width=end - start, left=start, height=0.3, color=colors(i),
                     alpha=0.7)
            if first_site <= start <= last_site:
                plt.text(start, min_fit + 1.1 + (i % 2) * 0.3, name, ha='left', va='center', fontsize=10)
            elif first_site <= end <= last_site or (start < first_site and end > last_site):
                plt.text(first_site, min_fit + 1.1 + (i % 2) * 0.3, name, ha='left', va='center', fontsize=10)

        # Add known conserved regions
        _, conserved_regions = get_conserved_regions()
        conserved_regions = [(value[0], value[1], key) for key, value in conserved_regions.items()]
        for i, (start, end, name) in enumerate(conserved_regions):
            if ((start <= first_site <= end) or (start <= last_site <= end)) or (
                    (first_site <= start <= last_site) and (first_site <= end <= last_site)):
                plt.barh(y=min_fit + 1.7 + (i % 3) * 0.3, width=end - start, left=start, height=0.3,
                         color='orange', alpha=0.6)
                plt.text(max(start, first_site), min_fit + 1.7 + (i % 3) * 0.3, name, ha='left',
                         va='center', fontsize=10)

        # Get secondary structure information to write on the plot
        df_help = df_sec_str[(df_sec_str['nt_site'] >= first_site) & (df_sec_str['nt_site'] <= last_site)]
        sites_help = df_help['nt_site'].values
        unpaired = (df_help['unpaired'].values == 0)
        paired = 1 - unpaired

        # Read in 21J reference sequence, TODO: Find a better solution that not only works for 21J
        with open(f'21J_refseq', 'r') as file:
            ref_seq = file.read()

        # Get corresponding section of the reference sequence
        nt_seq = ref_seq[int(first_site - 1):int(last_site)]

        # Add parent nucleotides to plot
        sites_help = np.arange(first_site, last_site + 1)
        for i, letter in enumerate(nt_seq):
            plt.text(sites_help[i], min_fit + 0.2, letter, ha='center', va='bottom', fontsize=11)

        # Add pairing information to plot
        plt.bar(sites_help, unpaired * 0.3, bottom=min_fit + 0.6, color='red', alpha=0.7)
        plt.bar(sites_help, paired * 0.3, bottom=min_fit + 0.6, color='blue', alpha=0.7)

        # Cross every dot in the scatter plot which has zero counts
        actual_counts = df['actual_count'].values
        zero_count_sites = nt_positions[actual_counts == 0]
        zero_count_fitness = fitness_estimates['corr'][actual_counts == 0]
        plt.scatter(zero_count_sites, zero_count_fitness, color='black', marker='x', alpha=0.5, label='0 counts')

        # Add every 10th nucleotide position as x-tick
        sites = probabilistic_pos[(probabilistic_pos >= first_site) & (probabilistic_pos <= last_site)]
        xticks = range(min(sites) + 10 - min(sites) % 10, max(sites), 10)
        plt.xticks(xticks)

        # Add legend for mutation type colors
        legend_handles = [mpatches.Patch(color=color, label=mut_type) for mut_type, color in mut_type_cols_2.items()]
        plt.legend(handles=legend_handles, fontsize='xx-small', loc='lower right')

        # Add vertical gridlines
        plt.minorticks_on()
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
        plt.grid(which='major', axis='x', linestyle='-', linewidth=0.5, color='black')
        plt.grid(which='minor', axis='x', linestyle=':', linewidth=0.5, color='gray')

        # Add legend and show plot
        plt.tight_layout()
        plt.show()


def make_region_list(arr, min_d, max_size):
    regions = []
    start = arr[0]
    for i, site in enumerate(arr[:-1]):
        if (arr[i + 1] > site + min_d) and (arr[i + 1] - start > max_size):
            end = site
            regions.append((start, end))
            start = arr[i + 1]
    regions.append((start, arr[-1]))
    return regions


def get_outlier_regions(fitness_estimates, outlier_condition, min_distance, max_region_size):
    # Get mean and standard deviation of fitness estimates
    mean = fitness_estimates.mean()
    std = fitness_estimates.std()

    # Find outliers
    outliers_above = np.where((fitness_estimates > mean + outlier_condition * std))[0] + 265
    outliers_below = np.where((fitness_estimates < mean - outlier_condition * std))[0] + 265

    # Make list of region boundaries
    outlier_regions_above = make_region_list(outliers_above, min_distance, max_region_size)
    outlier_regions_below = make_region_list(outliers_below, min_distance, max_region_size)

    return outlier_regions_below, outlier_regions_above


def get_min_max_fitness_across_regions(clade, regions):
    # Load mutation counts
    df = load_mut_counts(clade, mut_types='non-excluded')
    df = add_predictions(df.copy(), clade)

    min_fit, max_fit = 1000, -1000
    for region in regions:
        df_region = df[(df['nt_site'] <= region[1]) & (df['nt_site'] >= region[0])]
        fitness = np.log(df_region['actual_count'] + 0.5) - df['pred_log_count']
        min_fit, max_fit = min(min_fit, np.min(fitness)), max(max_fit, np.max(fitness))

    return min_fit, max_fit


def plot_gene_boundaries(first_site, last_site, min_fit, color_list, y_offset=1.1):
    for i, (start, end, name) in enumerate(gene_boundaries):
        plt.barh(y=min_fit + y_offset + (i % 2) * 0.3, width=end - start, left=start, height=0.3, color=color_list(i),
                 alpha=0.7)
        if first_site <= start <= last_site:
            plt.text(start, min_fit + y_offset + (i % 2) * 0.3, name, ha='left', va='center', fontsize=10)
        elif first_site <= end <= last_site or (start < first_site and end > last_site):
            plt.text(first_site, min_fit + y_offset + (i % 2) * 0.3, name, ha='left', va='center', fontsize=10)


def plot_conserved_regions(first_site, last_site, min_fit, conserved_regions, y_offset=1.7):
    for i, (start, end, name) in enumerate(conserved_regions):
        if ((start <= first_site <= end) or (start <= last_site <= end)) or (
                (first_site <= start <= last_site) and (first_site <= end <= last_site)):
            plt.barh(y=min_fit + y_offset + (i % 3) * 0.3, width=end - start, left=start, height=0.3,
                     color='orange', alpha=0.6)
            plt.text(max(start, first_site), min_fit + y_offset + (i % 3) * 0.3, name, ha='left', va='center',
                     fontsize=10)


def plot_secondary_structure(first_site, last_site, min_fit, nt_seq):
    df_help = df_sec_str[(df_sec_str['nt_site'] >= first_site) & (df_sec_str['nt_site'] <= last_site)]
    sites_help = df_help['nt_site'].values
    unpaired = (df_help['unpaired'].values == 0)
    paired = 1 - unpaired

    sites_help = np.arange(first_site, last_site + 1)
    for i, letter in enumerate(nt_seq):
        plt.text(sites_help[i], min_fit + 0.2, letter, ha='center', va='bottom', fontsize=11)

    plt.bar(sites_help, unpaired * 0.3, bottom=min_fit + 0.6, color='red', alpha=0.7)
    plt.bar(sites_help, paired * 0.3, bottom=min_fit + 0.6, color='blue', alpha=0.7)


def add_xticks(probabilistic_pos, first_site, last_site):
    sites = probabilistic_pos[(probabilistic_pos >= first_site) & (probabilistic_pos <= last_site)]
    xticks = range(min(sites) + 10 - min(sites) % 10, max(sites), 10)
    plt.xticks(xticks)


def plot_fitness_estimates(df_noncod, df_nonexc, probabilistic_pos, probabilistic_estims, original_pos, original_estims,
                           hyperparams, selected_trace, first_site, last_site, min_size=60, min_fit=-7, max_fit=5):
    # Enlarge region if too small
    if last_site - first_site < min_size:
        diff = min_size - last_site + first_site
        first_site -= int(diff / 2)
        last_site += int(diff / 2)

    print(f"region {first_site}-{last_site}")

    # Define size of figures
    w = 0.125 * (last_site - first_site)
    h = 5

    # Get mean and st.dev of probabilistic and naive fitness estimate across all sites present in the above data
    mean_new, stdev_new = np.average(probabilistic_estims[selected_trace]), np.std(probabilistic_estims[selected_trace])
    mean_naive, stdev_naive = np.average(original_estims), np.std(original_estims)

    # Create first figure
    plt.figure(figsize=(w, h))

    # Add means and st.devs.
    plt.axhline(0, linestyle='-', color='gray', lw=2, alpha=0.5)
    plt.axhline(mean_new, color='green', linestyle='-', alpha=0.3 + selected_trace * 0.3)
    plt.axhline(mean_new + OUTLIER_COND * stdev_new, color='green', linestyle='--', alpha=0.3 + selected_trace * 0.3)
    plt.axhline(mean_new - OUTLIER_COND * stdev_new, color='green', linestyle='--', alpha=0.3 + selected_trace * 0.3)
    plt.axhline(mean_naive, color='red', linestyle='-', alpha=0.5)
    plt.axhline(mean_naive + OUTLIER_COND * stdev_naive, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    plt.axhline(mean_naive - OUTLIER_COND * stdev_naive, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

    # Add probabilistic fitness estimates
    for i in range(len(hyperparams)):
        plt.plot(probabilistic_pos, probabilistic_estims[i], linestyle='-', marker='o',
                 label=fr"$\tau=${hyperparams[i]['sigma_n']}, $\rho_{{i}}=${hyperparams[i]['sigma_f'][0]}",
                 markersize=3, linewidth=2, color='green', alpha=0.3 + i * 0.3)

    # Add naive fitness estimate
    plt.plot(original_pos, original_estims, linestyle='-', marker='o', label="naive estimate",
             markersize=3, linewidth=1.5, color='red', alpha=0.5)

    # Set limits of plot
    plt.xlim((first_site, last_site))
    plt.ylim((min_fit, max_fit))

    # Add axes labels
    plt.xlabel('nucleotide position')
    plt.ylabel('fitness effect')

    # Add gene boundaries
    colors = cm.get_cmap('tab20', len(gene_boundaries))
    plot_gene_boundaries(first_site, last_site, min_fit, colors)

    # Add conserved regions
    _, conserved_regions = get_conserved_regions()
    conserved_regions = [(value[0], value[1], key) for key, value in conserved_regions.items()]
    plot_conserved_regions(first_site, last_site, min_fit, conserved_regions)

    # Add reference sequence
    with open(f'21J_refseq', 'r') as file:
        ref_seq = file.read()
    nt_seq = ref_seq[int(first_site - 1):int(last_site)]
    plot_secondary_structure(first_site, last_site, min_fit, nt_seq)

    # Add x-ticks and grid
    add_xticks(probabilistic_pos, first_site, last_site)
    plt.minorticks_on()
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
    plt.grid(which='major', axis='x', linestyle='-', linewidth=0.5, color='black')
    plt.grid(which='minor', axis='x', linestyle=':', linewidth=0.5, color='gray')

    # Add legend and show plot
    plt.legend(fontsize=9)
    plt.tight_layout()

    # Save figure
    save_dir = f'results/{first_site}_{last_site}'
    os.makedirs(save_dir, exist_ok=True)
    save_filename = 'empty.png'
    save_path = os.path.join(save_dir, save_filename)
    plt.savefig(save_path)
    plt.close()

    # Also create plots with fitness estimates of individual mutations
    for counter, (df, colors) in enumerate([(df_noncod, mut_type_cols), (df_nonexc, mut_type_cols_2)]):
        # Create figure
        plt.figure(figsize=(w, h))

        # Add additional information
        gene_colors = cm.get_cmap('tab20', len(gene_boundaries))
        plot_gene_boundaries(first_site, last_site, min_fit, gene_colors)
        plot_conserved_regions(first_site, last_site, min_fit, conserved_regions)
        plot_secondary_structure(first_site, last_site, min_fit, nt_seq)

        # Add means and st.devs.
        plt.axhline(0, linestyle='-', color='gray', lw=2, alpha=0.5)
        plt.axhline(mean_new, color='green', linestyle='-', alpha=0.3 + selected_trace * 0.3)
        plt.axhline(mean_new + OUTLIER_COND * stdev_new, color='green', linestyle='--',
                    alpha=0.3 + selected_trace * 0.3)
        plt.axhline(mean_new - OUTLIER_COND * stdev_new, color='green', linestyle='--',
                    alpha=0.3 + selected_trace * 0.3)
        plt.axhline(mean_naive, color='red', linestyle='-', alpha=0.5)
        plt.axhline(mean_naive + OUTLIER_COND * stdev_naive, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        plt.axhline(mean_naive - OUTLIER_COND * stdev_naive, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

        # Select data that falls into this region
        df = df[(df['nt_site'] >= first_site) & (df['nt_site'] <= last_site)]
        nt_positions = df['nt_site'].values
        fitness_estimates = np.log(df['actual_count'].values + 0.5) - df['pred_log_count'].values

        if counter == 1:
            # Color datapoints according to mutation type
            colores = np.full(len(df), "", dtype='<U20')
            colores[df['noncoding'].values == True] = "noncoding"
            colores[df['synonymous'].values == True] = "synonymous"
            colores[colores == ''] = 'nonsynonymous'
            scatter_colors = np.vectorize(colors.get)(colores)

            # Add scatter plot
            plt.scatter(nt_positions, fitness_estimates, c=scatter_colors, s=50, edgecolors='black', alpha=0.8)
        else:
            # Color datapoints according to mutation type
            mut_type = df['nt_mutation'].apply(lambda x: x[0] + x[-1])
            scatter_colors = mut_type.apply(lambda x: colors[x])

            # Add scatter plot
            plt.scatter(nt_positions, fitness_estimates, c=scatter_colors, s=50, edgecolors='black', alpha=0.8)

        # Add legend for mutation type colors
        legend_handles = [mpatches.Patch(color=color, label=mut_type) for mut_type, color in colors.items()]
        plt.legend(handles=legend_handles, fontsize='xx-small', loc='upper right', ncol=2-counter)

        # Add x-ticks and vertical grid
        add_xticks(probabilistic_pos, first_site, last_site)
        plt.minorticks_on()
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
        plt.grid(which='major', axis='x', linestyle='-', linewidth=0.5, color='black')
        plt.grid(which='minor', axis='x', linestyle=':', linewidth=0.5, color='gray')

        # Add probabilistic fitness estimates
        i = selected_trace
        plt.plot(probabilistic_pos, probabilistic_estims[i], linestyle='-', marker='o',
                 markersize=3, linewidth=2, color='green', alpha=0.3 + i * 0.3)

        # Add naive fitness estimate
        plt.plot(original_pos, original_estims, linestyle='-', marker='o', label="naive estimate",
                 markersize=3, linewidth=1.5, color='red', alpha=0.5)

        # Set x- and y-limits
        plt.xlim((first_site, last_site))
        plt.ylim((min_fit, max_fit))

        # Add axis labels
        plt.xlabel('nucleotide position')
        plt.ylabel('fitness effect')

        # Add legend and show plot
        plt.tight_layout()
        save_filename = f'{["noncoding", "all"][counter]}.png'
        save_path = os.path.join(save_dir, save_filename)
        plt.savefig(save_path)
        plt.close()


if __name__ == '__main__':

    # Define hyperparameters for probabilistic fitness estimate
    hyperparams = [{'sigma_n': 0.1, 'sigma_s': 1000, 'sigma_f': [0.1, 10000]},
                   {'sigma_n': 0.1, 'sigma_s': 1000, 'sigma_f': [0.2, 10000]},
                   {'sigma_n': 0.1, 'sigma_s': 1000, 'sigma_f': [0.3, 10000]}]

    # Compute naive and probabilistic fitness estimate at all sites of the genome except excluded ones at start and end
    sites_probabilistic, estimates_probabilistic, sites_naive, estimates_naive = get_probabilistic_fitness_estimates(
        '21J', hyperparams)

    # Find outlier regions to zoom into
    OUTLIER_COND = 3  # in terms of standard deviation of the selected probabilistic fitness estimate
    MIN_DISTANCE = 10  # minimal distance for two points to be considered as belonging to different regions
    SELECTED_TRACE = 1  # determines which of the hyperparameters is used for the outlier detection
    low_outlier_regions, high_outlier_regions = get_outlier_regions(estimates_probabilistic[SELECTED_TRACE],
                                                                    outlier_condition=OUTLIER_COND,
                                                                    min_distance=MIN_DISTANCE, max_region_size=100)

    # Get minimal and maximal fitness in all the zoom-in plots
    MIN_FIT, MAX_FIT = get_min_max_fitness_across_regions(clade='21J', regions=low_outlier_regions)

    # Load mutation data to be plotted in the zoom-in plots
    df_noncod = load_mut_counts(clade='21J', include_noncoding='True', include_tolerant_orfs='True',
                                remove_orf9b=True, incl_orf1ab_overlap=None)
    df_noncod = add_predictions(df_noncod.copy(), clade='21J')
    df_nonexc = load_mut_counts(clade='21J', mut_types='non-excluded')
    df_nonexc = add_predictions(df_nonexc.copy(), clade='21J')

    # Create three different zoom-in plots for every detected outlier region
    for region in low_outlier_regions:
        plot_fitness_estimates(df_noncod, df_nonexc, sites_probabilistic, estimates_probabilistic, sites_naive,
                                 estimates_naive, hyperparams, selected_trace=SELECTED_TRACE,
                                 first_site=region[0] - MIN_DISTANCE, last_site=region[1] + MIN_DISTANCE,
                                 min_fit=MIN_FIT, max_fit=MAX_FIT)
