import numpy as np
from matplotlib import pyplot as plt
from helper import load_mut_counts
from general_linear_models.fit_general_linear_models import add_predictions


def a_k_minus_p(k, p):

    if k - p == 1:
        a = 0.5 * (1 / sigma_N ** 2 + 1 / sigma_S ** 2 + 1 / sigma_F ** 2)
    else:
        a = 0.5 * (1 / sigma_F ** 2 + 2 / sigma_N ** 2 + 1 / sigma_S ** 2 - 1 / (sigma_N ** 2 + sigma_S ** 2) - 1 / (2 * sigma_N ** 4 * a_k_minus_p(k, p + 1)))
    return a


def b_k_minus_p(k, p):

    if k - p == 1:
        b = - alpha[1] / sigma_F ** 2
    else:
        b = - alpha[k - 1] / sigma_F ** 2 + b_k_minus_p(k, p + 1) / (2 * sigma_N ** 2 * a_k_minus_p(k, p + 1))
    return b


def c_k_minus_p(k, p):

    if k - p == 1:
        c = - alpha[1] ** 2 / (2 * sigma_F ** 2)
    else:
        c = c_k_minus_p(k, p + 1) - b_k_minus_p(k, p + 1) ** 2 / (2 * sigma_N ** 2 * a_k_minus_p(k, p + 1)) + alpha[k - 1] ** 2 / (2 * sigma_F ** 2)
    return c


def a_k_plus_p(k, p):

    if p == 1:
        a = 0.5 * (2 / sigma_N ** 2 + 1 / sigma_S ** 2 + 1 / sigma_F ** 2 - 1 / (sigma_N ** 2 + sigma_S ** 2))
    else:
        a = 0.5 * (2 / sigma_N ** 2 + 1 / sigma_S ** 2 + 1 / sigma_F ** 2 - 1 / (sigma_N ** 2 + sigma_S ** 2) - 1 / (2 * sigma_N ** 4 * a_k_plus_p(k, p - 1)))

    return a


def b_k_plus_p(k, p, f_k):
    if p == 1:
        b = - alpha[k + 1] / sigma_F ** 2 - f_k / sigma_N ** 2
    else:
        b = - alpha[k + 2] / sigma_F ** 2 + b_k_plus_p(k, p - 1, f_k) / (2 * a_k_plus_p(k, p - 1) * sigma_N ** 2)

    return b


def c_k_plus_p(k, p, f_k):

    if p == 1:
        c = - alpha[k + 1] ** 2 / (2 * sigma_F ** 2)
    else:
        c = c_k_plus_p(k, p - 1, f_k) - b_k_plus_p(k, p - 1, f_k) ** 2 / (4 * a_k_plus_p(k, p - 1)) + alpha[k + 2] ** 2 / (2 * sigma_F ** 2)

    return c


def p_f_k(f_k, k):
    """
    :param f_k: fitness effect at site k
    :param k: site
    :param alpha: array with observed fitness effects at every site
    :return:
    """

    msg_left = np.exp((b_k_minus_p(k, 1) - f_k / sigma_N ** 2) ** 2 / (4 * a_k_minus_p(k, 1)))

    msg_right = np.exp(b_k_plus_p(k, N - k, f_k) ** 2 / (4 * a_k_plus_p(k, N - k)) - c_k_plus_p(k, N - k, f_k))

    left_over = np.exp(- (f_k ** 2 / 2) * (2 / sigma_N ** 2 + 1 / sigma_S ** 2 + 1 / (sigma_N ** 2 + sigma_S ** 2)))

    p_n_given_f_k = np.exp(- (alpha[k] - f_k) ** 2 / (2 * sigma_F ** 2))

    return msg_left * msg_right * left_over * p_n_given_f_k


if __name__ == '__main__':

    CLADE = '21J'

    sigma_N = 1
    sigma_S = 1
    sigma_F = 1

    # Load all mutation counts and add prediction
    df = load_mut_counts(clade=CLADE, mut_types='all')
    df = add_predictions(df.copy(), clade=CLADE)

    # Remove excluded sites at the beginning and end (new length: 29674 - 265 = 29409)
    df = df[(df['nt_site'] > 265) & (df['nt_site'] < 29675)]

    # Remove and remember all sites that have excluded mutations
    nt_sites_with_exclusions = df.groupby('nt_site')['exclude'].any()
    nt_sites_with_exclusions = nt_sites_with_exclusions[nt_sites_with_exclusions].index.tolist()
    df = df[~df['nt_site'].isin(nt_sites_with_exclusions)]

    # Average fitness effect at every site
    fitness = np.log(df['actual_count'].values + 0.5) - df['pred_log_count'].values
    alpha = (fitness[0:-2:3] + fitness[1:-1:3] + fitness[2::3]) / 3

    # Length of sequence without excluded sites
    N = len(alpha)

    # Plot p_f_k
    f = np.arange(-1, 1, 0.05)
    p_f_k_vec = np.vectorize(p_f_k)
    p = p_f_k_vec(f, k=1000)

    plt.plot(f, p)
    plt.show()
