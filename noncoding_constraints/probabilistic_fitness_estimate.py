import numpy as np
import pandas as pd
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from helper import load_mut_counts, gene_boundaries, tolerant_orfs
from general_linear_models.fit_general_linear_models import add_predictions
from Bio import SeqIO


plt.rcParams.update({'font.size': 12})

# Load secondary structure information, TODO: The illustration in the experimental paper is for Vero cells
df_sec_str = pd.read_csv(
    f'/Users/georgangehrn/Desktop/SARS-CoV2-mut-fitness/sec_stru_data/data/sec_structure_Huh7.txt',
    header=None, sep='\s+').drop(columns=[2, 3, 5])
df_sec_str.rename(columns={0: 'nt_site', 1: 'nt_type', 4: 'unpaired'}, inplace=True)


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
    msg_left = (b_left + f_vals*inv_var_n)**2 / (4*a_left) - 0.5 * f_vals**2 * (inv_var_s + inv_var_n)
    msg_right = ((b_right + f_vals*inv_var_n)**2 / (4*a_right)) - 0.5 * f_vals**2 * (inv_var_n - 1/(var_n + var_s))
    p_n_given_f_k = - (f_hat - f_vals)**2 / (2 * sigma_f**2)

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

    var_n = sigma_n**2
    var_s = sigma_s**2
    var_f = sigma_f**2

    inv_var_n = 1 / var_n
    inv_var_s = 1 / var_s
    inv_var_f = 1 / var_f

    a = np.zeros(len(f_hat))
    b = np.zeros(len(f_hat))

    a[0] = 0.5 * (inv_var_n + inv_var_s + inv_var_f[0] - 1/(var_n + var_s))
    b[0] = f_hat[0] * inv_var_f[0]

    for i in range(1, len(f_hat)):

        a[i] = 0.5 * (2*inv_var_n + inv_var_f[i] + inv_var_s - 1/(var_n + var_s) - 1/(2*a[i-1]*var_n**2))
        b[i] = b[i-1]/(2*a[i-1]*var_n) + f_hat[i]*inv_var_f[i]

    return a, b


def get_right_msgs(f_hat, sigma_n, sigma_s, sigma_f):

    var_n = sigma_n**2
    var_s = sigma_s**2
    var_f = sigma_f**2

    inv_var_n = 1 / var_n
    inv_var_s = 1 / var_s
    inv_var_f = 1 / var_f

    a = np.zeros(len(f_hat))
    b = np.zeros(len(f_hat))

    a[-1] = 0.5 * (inv_var_n + inv_var_s + inv_var_f[-1])
    b[-1] = f_hat[-1] * inv_var_f[-1]

    for i in range(2, len(f_hat)+1):

        a[-i] = 0.5 * (2*inv_var_n + inv_var_f[-i] + inv_var_s - 1/(var_n + var_s) - 1/(2*a[-i+1]*var_n**2))
        b[-i] = b[-i+1]/(2*a[-i+1]*var_n) + f_hat[-i]*inv_var_f[-i]

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
    return original_positions, new_fitness_estimates, original_positions[~coding_mask], original_fitness_estimates[~coding_mask]


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
    inv_tau_sq = 1 / tau**2

    # Prepare tridiagonal coefficient matrix
    diagonal = 1/sigma**2 + 1/rho**2 + np.append(0, inv_tau_sq[:-1]) + np.append(inv_tau_sq[:-1], 0)
    superdiagonal = - inv_tau_sq[:-1]
    subdiagonal = - inv_tau_sq[:-1]

    # Prepare y in the linear system of equations Af = y
    y = g / rho ** 2

    # Solve linear system of equations to get probabilistic fitness estimates
    f = solve_tridiagonal_lse(diagonal, superdiagonal, subdiagonal, y)

    return f


def get_probabilistic_fitness_estimates(clade, hyperparams):
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
    mask_orf9b = ((df['gene'] == 'N;ORF9b') & (df['clade_founder_aa'].apply(lambda x: x[0]) == df['mutant_aa'].apply(lambda x: x[0]))).values
    # Mask for sites which are in stop codon tolerant open reading frames
    pattern = '|'.join(tolerant_orfs)
    mask_tolerant = (df['gene'].str.contains(pattern)).values
    # Mask for non-excluded sites
    mask_nonexcluded = ~df['exclude'].values
    # Combine all masks
    df['mask'] = (mask_noncoding + mask_synonymous + mask_orf9b + mask_tolerant) * mask_nonexcluded

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

    # Set sigma_i (cf. notes)
    sigma_i = np.full((L,), hyperparams['sigma_s'])

    # Set rho_i (cf. notes)
    rho_i = np.full((L,), hyperparams['sigma_f'][0])
    mask_rho = ~df.groupby('nt_site')['mask'].any()
    rho_i[mask_rho] = hyperparams['sigma_f'][1]

    # Set tau_i (cf. notes)
    tau_i = np.full((L,), hyperparams['sigma_n'])

    # Get probabilistic fitness estimates
    f_i = solve_lse(sigma_i, rho_i, tau_i, g_i)

    # nucleotide positions
    nucleotide_positions = np.arange(266, 29674 + 1)

    # Return nucleotide sites >265 and <29675, the corresponding probabilistic fitness estimates, and the naive
    # fitness estimate at all sites which have at least one included mutation
    return nucleotide_positions, f_i, nucleotide_positions[~mask_rho], g_i[~mask_rho]


def plot_fitness_estimates(probabilistic_pos, probabilistic_estims, original_pos, original_estims, hyperparams, first_site, last_site, min_fit=-7):
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
        # Get average fitness estimates and standard deviations across all sites present in the respective inputs
        mean_new, stdev_new = np.average(probabilistic_estims), np.std(probabilistic_estims)
        mean_old, stdev_old = np.average(original_estims), np.std(original_estims)

        # Increase figure size if too small
        if last_site - first_site < 60:
            diff = 60 - last_site + first_site
            first_site = first_site - int(diff/2)
            last_site = last_site + int(diff / 2)

        # Define figure size
        w = 0.125 * (last_site - first_site)
        h = 6
        plt.figure(figsize=(w, h), dpi=300)

        # Add mean and standard deviations
        plt.axhline(0, linestyle='-', color='gray', lw=2, alpha=0.5)
        plt.axhline(mean_new, color='green', linestyle='-')
        plt.axhline(mean_new + OUTLIER_COND * stdev_new, color='green', linestyle='--')
        plt.axhline(mean_new - OUTLIER_COND * stdev_new, color='green', linestyle='--')
        plt.axhline(mean_old, color='red', linestyle='-', alpha=0.5)
        plt.axhline(mean_old + OUTLIER_COND * stdev_old, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        plt.axhline(mean_old - OUTLIER_COND * stdev_old, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

        # Plot both fitnesss estimates
        plt.plot(probabilistic_pos, probabilistic_estims, linestyle='-', marker='o', label="probabilistic estimate", markersize=3,
                 linewidth=2, color='green', alpha=0.5)
        plt.plot(original_pos, original_estims, linestyle='-', marker='o', label="naive estimate",
                 markersize=3, linewidth=2, color='red', alpha=0.5)

        # Set limits of plot
        plt.xlim((first_site, last_site))
        plt.ylim((min_fit, 5))

        # Add labels and title
        plt.xlabel('nucleotide position')
        plt.ylabel('fitness effect')
        plt.title(f"$\sigma_{{n}}=${hyperparams['sigma_n']}, $\sigma_{{S}}=${hyperparams['sigma_s']}, $\sigma_{{F}}=${hyperparams['sigma_f']}")

        # Add gene boundaries to plot, TODO: Make all of this a reusable function
        colors = cm.get_cmap('tab20', len(gene_boundaries))
        for i, (start, end, name) in enumerate(gene_boundaries):
            plt.barh(y=min_fit + 1.1 + (i % 2) * 0.3, width=end - start, left=start, height=0.3, color=colors(i), alpha=0.7)
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
            plt.text(sites_help[i], min_fit + 0.2, letter, ha='center', va='bottom')

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
    outliers = np.where((fitness_estimates < mean - outlier_condition * std) | (fitness_estimates > mean + outlier_condition * std))[0] + 265

    # Make list of region boundaries
    outlier_regions = make_region_list(outliers, min_distance, max_region_size)

    return outlier_regions


if __name__ == '__main__':

    # Define hyperparameters for probabilistic fitness estimate
    hyperparams = [{'sigma_n': 0.1, 'sigma_s': 10, 'sigma_f': [0.01, 10000]},
                   {'sigma_n': 0.1, 'sigma_s': 10, 'sigma_f': [0.1, 10000]},
                   {'sigma_n': 0.1, 'sigma_s': 10, 'sigma_f': [0.2, 10000]}][2]

    # Compute naive and probabilistic fitness estimate at all sites of the genome except excluded ones at start and end
    sites_probabilistic, estimates_probabilistic, sites_naive, estimates_naive = get_probabilistic_fitness_estimates('21J', hyperparams)

    # Find outlier regions to zoom into
    OUTLIER_COND = 3
    outlier_regions = get_outlier_regions(estimates_probabilistic, outlier_condition=OUTLIER_COND, min_distance=10, max_region_size=120)

    # Plot new vs. old fitness estimate within a defined window
    for region in outlier_regions[:10]:
        plot_fitness_estimates(sites_probabilistic, estimates_probabilistic, sites_naive, estimates_naive, hyperparams, first_site=region[0]-10, last_site=region[1]+10)
