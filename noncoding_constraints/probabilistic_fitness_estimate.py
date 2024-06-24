import time
import numba
import numpy as np
from matplotlib import pyplot as plt
from helper import load_mut_counts
from general_linear_models.fit_general_linear_models import add_predictions

plt.rcParams.update({'font.size': 12})


def log_p_of_f_k(sites, f_hat, f_vals, a_left, b_left, a_right, b_right, sigma_n, sigma_s, sigma_f):
    """
    :param sites: nucleotide sites k for which log(p(f_k)) is calculated, dimension (n_sites,)
    :param f_vals: fitness values for which log(p(f_k)) is calculated, dimension (n_f,)
    :return: log(p(f_k)) for all sites and fitness values, dimension (n_sites, n_f)
    """

    var_n = sigma_n ** 2
    var_s = sigma_s ** 2
    var_f = sigma_f ** 2

    inv_var_n = 1 / var_n
    inv_var_s = 1 / var_s
    inv_var_f = 1 / var_f

    # Get left and right messages, all of dimension (n_sites,)
    a_left = a_left[sites - 1]
    b_left = b_left[sites - 1]
    a_right = a_right[sites - 1]
    b_right = b_right[sites - 1]

    # Put everything together
    msg_left = ((b_left + f_vals.reshape(-1, 1)*inv_var_n)**2 / (4*a_left)).T - 0.5*f_vals.reshape(1, -1)*(inv_var_s + inv_var_n)
    msg_right = ((b_right + f_vals.reshape(-1, 1)*inv_var_n)**2 / (4*a_right)).T - 0.5*f_vals.reshape(1, -1)*(inv_var_n - 1/(var_n + var_s))
    p_n_given_f_k = (- (f_hat.reshape(-1, 1) - f_vals)**2 / (2 * sigma_f.reshape(-1, 1)**2))

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

    for i in range(2, len(f_hat)):

        a[-i] = 0.5 * (2*inv_var_n + inv_var_f[-i] + inv_var_s - 1/(var_n + var_s) - 1/(2*a[-i+1]*var_n**2))
        b[-i] = b[-i+1]/(2*a[-i+1]*var_n) + f_hat[-i]*inv_var_f[-i]

    return a, b


if __name__ == '__main__':

    # Define clade to take data from
    CLADE = '21J'

    # Load all mutation counts and add predictions
    df = load_mut_counts(clade=CLADE, mut_types='all')
    df = add_predictions(df.copy(), clade=CLADE)

    # Remove excluded sites at the beginning and the end of the genome (new length: 29674 - 265 = 29409)
    df = df[(df['nt_site'] > 265) & (df['nt_site'] < 29675)]

    # Remove and remember all other sites that have any excluded mutations
    sites_with_exclusions = df.groupby('nt_site')['exclude'].any()
    sites_with_exclusions = sites_with_exclusions[sites_with_exclusions].index.values
    df = df[~df['nt_site'].isin(sites_with_exclusions)]

    # Get average fitness effect (f_hat, cf. notes) at all sites that have no excluded mutations, dimension (N,)
    f_hat = np.log(df['actual_count'].values + 0.5) - df['pred_log_count'].values
    f_hat = (f_hat[0:-2:3] + f_hat[1:-1:3] + f_hat[2::3]) / 3

    # Length of new reference genome
    N = len(f_hat)

    # Define hyperparameters
    sigma_n = 0.1
    sigma_s = 2
    sigma_f = np.full((N,), 0.1)

    # Compute coefficients from iterative calculation of right/left messages, each of dimension (N,)
    a_left, b_left = get_left_msgs(f_hat, sigma_n, sigma_s, sigma_f)
    a_right, b_right = get_right_msgs(f_hat, sigma_n, sigma_s, sigma_f)

    # Define window in terms of the original coordinate system
    start, end = 25000, 26000

    # original and translated (new) positions of non-excluded sites within the defined window
    original_positions = np.arange(start, end + 1)
    mask = ~np.isin(original_positions, sites_with_exclusions)
    original_positions = original_positions[mask]
    new_positions = translate_position(sites_with_exclusions, original_positions)

    # Get original, non-probabilistic fitness estimates
    original_fitness_estimates = f_hat[new_positions - 1]

    # Define fitness values to sweep over (x-axis of optimization task)
    f_values = np.arange(-3, 2, 0.2)

    # Calculate quantity proportional to the log probability
    # new_positions is of dimension (n_sites,), f_values is of dimension (n_f,)
    log_p_values = log_p_of_f_k(new_positions, original_fitness_estimates, f_values, a_left, b_left, a_right, b_right, sigma_n, sigma_s, sigma_f[new_positions - 1])

    # Get most likely fitness value at every site in the window
    # f_values is of dimension (n_f,), log_p_values is of dimension (n_sites, n_f)
    new_fitness_estimates = get_most_likely_f_value(f_values, log_p_values)

    # Plot estimated fitness effects across region, TODO: Formulate as function
    w = 0.125 * (end - start)
    max_fit = max(np.max(new_fitness_estimates), np.max(original_fitness_estimates))
    h = 0.6 * (max_fit + 7.5)
    plt.figure(figsize=(w, h))
    plt.plot(original_positions, new_fitness_estimates, linestyle='-', marker='o', label="new", markersize=3, linewidth=1.5)
    plt.plot(original_positions, original_fitness_estimates, linestyle='-', marker='o', label="old", markersize=3, linewidth=1.5)
    plt.xlim((start, end))
    plt.ylim((-7.5, max_fit))
    plt.title(f"$\sigma_{{N}}=${sigma_n}, $\sigma_{{S}}=${sigma_s}, $\sigma_{{F}}=${sigma_f[0]}")
    plt.xlabel('nucleotide position')
    plt.ylabel('fitness effect')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
