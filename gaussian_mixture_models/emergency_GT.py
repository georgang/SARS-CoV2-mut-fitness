import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix

plt.rcParams.update({'font.size': 14})


def gmm_fit_plot_1d(df, nt_x, nt_y, pseudo_count):
    # Extract counts and log_counts
    counts = df['actual_count'].values
    log_counts = np.log(counts + pseudo_count)

    # Calculate average log_count to determine initial means of GMM
    exp_log_count = np.average(log_counts)

    # Fit GMM to log(counts + 0.5)
    gmm = GaussianMixture(n_components=2, means_init=[[0.9 * exp_log_count], [1.1 * exp_log_count]],
                          n_init=5, random_state=1)
    gmm.fit(log_counts.reshape(-1, 1))

    # Prepare figure
    plt.figure(figsize=(8, 6), dpi=200)

    # Extract log counts at un-/paired sites separately
    log_counts_paired = log_counts[np.where(df['unpaired'].values == 0)]
    log_counts_unpaired = log_counts[np.where(df['unpaired'].values == 1)]

    w_paired = len(log_counts_paired) / (len(log_counts_paired) + len(log_counts_unpaired))
    w_unpaired = len(log_counts_unpaired) / (len(log_counts_paired) + len(log_counts_unpaired))

    min_log_count = np.min(log_counts)
    max_log_count = np.max(log_counts)

    hist_paired, bin_edges = np.histogram(log_counts_paired, range=(min_log_count, max_log_count),
                                          bins=60, density=True)
    hist_unpaired, bin_edges = np.histogram(log_counts_unpaired, range=(min_log_count, max_log_count),
                                            bins=60, density=True)
    full_hist = hist_paired * w_paired + hist_unpaired * w_unpaired

    w = bin_edges[1] - bin_edges[0]
    plt.bar(bin_edges[:-1], hist_paired * w_paired, width=w, alpha=0.6, color='blue', align='edge', label='paired')
    plt.bar(bin_edges[:-1], hist_unpaired * w_unpaired, width=w, alpha=0.6, color='red', align='edge', label='unpaired')
    plt.bar(bin_edges[:-1], full_hist, color='None', width=w, edgecolor='black', align='edge')

    # # Plot individual components of the Gaussian mixture model
    # x = np.linspace(min_log_count, max_log_count, 1000)
    # for i in range(2):
    #     mu = gmm.means_[i][0]
    #     sigma = np.sqrt(gmm.covariances_[i][0])
    #     weight = gmm.weights_[i]
    #     plt.plot(x, weight * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
    #              linewidth=3, color=['blue', 'red'][i])
    #
    # # Plot full pdf of the Gaussian mixture model
    # mixture_pdf = np.zeros_like(x)
    # for weight, mean, covariance in zip(gmm.weights_, gmm.means_, gmm.covariances_):
    #     mixture_pdf += weight * (1 / (np.sqrt(2 * np.pi * covariance[0, 0]))) * np.exp(
    #         -0.5 * ((x - mean[0]) ** 2 / covariance[0, 0]))
    # plt.plot(x, mixture_pdf, linewidth=3, color='black')

    # Compute confusion matrix
    predicted = gmm.predict(log_counts.reshape(-1, 1))
    ground_truth = df['unpaired'].values
    conf_mat = np.round(confusion_matrix(ground_truth, predicted, normalize='all'), 3)

    # Compute paired and unpaired mean from experimental data
    paired_counts = df['actual_count'].values[np.where(df['unpaired'].values == 0)]
    unpaired_counts = df['actual_count'].values[np.where(df['unpaired'].values == 1)]
    paired_mean = np.log(np.mean(paired_counts))
    unpaired_mean = np.log(np.mean(unpaired_counts))

    # Show plot with all components and text
    # plt.text(1.75, 0.1,
    #          'GMM means:\n' + str(round(gmm.means_[0, 0], 2)) + '\n' + str(round(gmm.means_[1, 0], 2))
    #          + '\n\nexp. data means:\n' + str(round(paired_mean, 2)) + '\n' + str(round(unpaired_mean, 2)), fontsize=12)
    # plt.text(1.75, 0.4, 'confusion matrix\n(0: paired, 1: unpaired):\n' + str(conf_mat) +
    #          '\n\nN = ' + str(len(log_counts)), fontsize=12)
    plt.title(nt_x + '->' + nt_y)
    plt.xlabel('ln(counts + 0.5)')
    plt.ylabel('# of sites (normalized)')
    plt.grid()
    plt.legend()
    plt.xlim((0, 6))
    plt.show()

    return gmm


if __name__ == '__main__':

    # Read file with mutation data
    df = pd.read_csv('counts/counts_all_21J_nonex_syn.csv')

    # Loop through all mutation types
    for nt_1 in ['A', 'C', 'G', 'T']:

        for nt_2 in ['A', 'C', 'G', 'T']:

            if nt_1 == 'A' and nt_2 == 'G':
                [nt_x, nt_y, cell_type, pseudo_count] = [nt_1, nt_2, 'Huh7', 0.5]

                df_mut = df[df['nt_mutation'].str.match('^' + nt_x + '.*' + nt_y + '$')]

                gmm_1d = gmm_fit_plot_1d(df_mut, nt_x, nt_y, pseudo_count)

                # actual_counts = df_mut['actual_count'].values
                # expected_counts = df_mut['expected_count'].values
                # n = len(actual_counts)
                # fit_uncorr = np.log((actual_counts + 0.5) / (expected_counts + 0.5))
                # [mean_uncorr, var_uncorr] = [np.mean(fit_uncorr), np.var(fit_uncorr)]
                #
                # fit_corr_2 = np.zeros(n)
                # for i in range(n):
                #     if df_mut['unpaired'].values[i] == 0:
                #         fit_corr_2[i] = np.log((actual_counts[i] + 0.5)) - gmm_1d.means_[0, 0] + 0.03
                #     else:
                #         fit_corr_2[i] = np.log((actual_counts[i] + 0.5)) - gmm_1d.means_[1, 0] + 0.03
                # [mean_corr_2, var_corr_2] = [np.mean(fit_corr_2), np.var(fit_corr_2)]
                #
                # plt.figure(figsize=(8, 6), dpi=200)
                #
                # plt.hist(fit_uncorr, bins=60, alpha=0.7, color='black', range=(-3 * var_uncorr, 3 * var_uncorr),
                #          density=True,
                #          label='uncorrected')
                # plt.text(-2.5, 0.6, 'mean = ' + str(round(mean_uncorr, 3)), color='black')
                # plt.text(-2.5, 0.5,
                #          'var = ' + str(round(var_uncorr, 3)), color='black')
                #
                # plt.hist(fit_corr_2, bins=60, alpha=0.7, color='green', range=(-3 * var_uncorr, 3 * var_uncorr),
                #          density=True,
                #          label='corrected')
                # plt.text(1.5, 0.6, 'mean = ' + str(round(mean_corr_2, 3)), color='green')
                # plt.text(1.5, 0.5, 'var = ' + str(round(var_corr_2, 3)), color='green')
                #
                # plt.xlabel('nt fitness effect')
                # plt.ylabel('# of sites (normalized)')
                # plt.title('fitness effect synonymous C->T\ncorrected by using GMM means')
                # plt.legend()
                # plt.grid()
                # plt.show()
