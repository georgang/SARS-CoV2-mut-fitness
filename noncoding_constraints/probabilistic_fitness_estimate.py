import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from helper import load_mut_counts, gene_boundaries, tolerant_orfs
from general_linear_models.fit_general_linear_models import add_predictions

plt.rcParams.update({'font.size': 12})


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

    # Load all mutation counts and add predictions
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
    # Mask for sites which are in the stop codon tolerant open reading frames
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
    # Aggregate counts of noncoding nonexcluded mutations
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

    # original and translated (new) positions of non-excluded sites
    original_positions = np.arange(266, 29674 + 1)

    return original_positions, f_i, original_positions[~mask_rho], g_i[~mask_rho]


def plot_fitness_estimates(probabilistic_pos, probabilistic_estims, original_pos, original_estims, hyperparams, first_site, last_site):
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

        # Define figure size
        w = 0.125 * (last_site - first_site)
        h = 6
        plt.figure(figsize=(w, h), dpi=300)

        # Add mean and standard deviations
        plt.axhline(0, linestyle='-', color='gray', lw=2, alpha=0.5)
        plt.axhline(mean_new, color='green', linestyle='-')
        plt.axhline(mean_new + stdev_new, color='green', linestyle='--')
        plt.axhline(mean_new - stdev_new, color='green', linestyle='--')
        plt.axhline(mean_old, color='red', linestyle='-', alpha=0.5)
        plt.axhline(mean_old + stdev_old, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        plt.axhline(mean_old - stdev_old, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

        # Plot both fitnesss estimates
        plt.plot(probabilistic_pos, probabilistic_estims, linestyle='-', marker='o', label="probabilistic estimate", markersize=3,
                 linewidth=2, color='green', alpha=0.5)
        plt.plot(original_pos, original_estims, linestyle='-', marker='o', label="original estimate",
                 markersize=3, linewidth=2, color='red', alpha=0.5)

        # Set limits of plot
        plt.xlim((first_site, last_site))
        plt.ylim((-7, 5))

        # Add labels and title
        plt.xlabel('nucleotide position')
        plt.ylabel('fitness effect')
        plt.title(f"$\sigma_{{N}}=${hyperparams['sigma_n']}, $\sigma_{{S}}=${hyperparams['sigma_s']}, $\sigma_{{F}}=${hyperparams['sigma_f']}")

        # Add gene boundaries to plot
        colors = cm.get_cmap('tab20', len(gene_boundaries))
        for i, (start, end, name) in enumerate(gene_boundaries):
            if start >= first_site and start <= last_site:
                plt.barh(y=-4.2 + (i % 2) * 0.2, width=end - start, left=start, height=0.2, color=colors(i), alpha=0.7)
                plt.text(start, -4.2 + (i % 2) * 0.2, name, ha='left', va='center', fontsize=10)
            elif end > first_site and end <= last_site:
                plt.barh(y=-4.2 + (i % 2) * 0.2, width=end - first_site, left=first_site, height=0.2, color=colors(i),
                         alpha=0.7)
                plt.text(first_site, -4.2 + (i % 2) * 0.2, name, ha='left', va='center', fontsize=10)

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


if __name__ == '__main__':

    # Define hyperparameters for probabilistic fitness estimate
    hyperparams = {'sigma_n': 0.1, 'sigma_s': 1, 'sigma_f': [0.3, 10000]}

    # Compute naive and probabilistic fitness estimate at all sites of the genome except excluded ones at start and end
    CLADE = '21J'
    sites_probabilistic, estimates_probabilistic, sites_naive, estimates_naive = get_probabilistic_fitness_estimates(CLADE, hyperparams)

    # Plot new vs. old fitness estimate within a defined window

    regions = [(13420, 13560), (21510, 21600), (25340, 25440), (26170, 26290), (26300, 26420), (26430, 26540),
    (27140, 27220), (28220, 28310)]
    #regions = [(28220, 28310)]

    for region in regions:
        plot_fitness_estimates(sites_probabilistic, estimates_probabilistic, sites_naive, estimates_naive, hyperparams, first_site=region[0], last_site=region[1])
