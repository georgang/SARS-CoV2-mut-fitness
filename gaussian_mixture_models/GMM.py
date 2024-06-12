import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix

plt.rcParams.update({'font.size': 14})

rna_dna_dict = {'G': 'G', 'C': 'C', 'A': 'A', 'T': 'U', 'U': 'T'}


def gmm_fit_plot_1d(df, nt_x, nt_y, exp_count, pseudo_count):

    # Extract counts from dataframe
    counts = df['actual_count'].values - pseudo_count

    # Extract counts at paired/unpaired sites separately
    counts_paired = counts[df['unpaired'].values == 0]
    counts_unpaired = counts[df['unpaired'].values == 1]

    [min, max] = [np.min(counts), np.max(counts)]

    [lb, ub] = [np.floor(np.percentile(counts, 0.25)), np.ceil(np.percentile(counts, 99))]

    n_bins = int(max-min)
    b = round(10*(ub - lb)/len(counts))
    if (max-min)%b != 0 and b != 0:
        max += b - (max-min)%b
        n_bins = int((max-min)/b)

    hist, bin_edges = np.histogram(counts, bins=n_bins, range=(min, max))
    hist_paired, bin_edges_paired = np.histogram(counts_paired, bins=n_bins, range=(min, max))
    hist_unpaired, bin_edges_unpaired = np.histogram(counts_unpaired, bins=n_bins, range=(min, max))
    plt.bar(bin_edges[:-1], hist, width=(bin_edges[-1]-bin_edges[0])/n_bins, alpha=0.7, align='edge', color='none', edgecolor='black')
    plt.bar(bin_edges_paired[:-1], hist_paired, width=(bin_edges[-1]-bin_edges[0])/n_bins,
            alpha=0.6, align='edge', color='blue')
    plt.bar(bin_edges_unpaired[:-1], hist_unpaired, width=(bin_edges[-1]-bin_edges[0])/n_bins,
            alpha=0.6, align='edge', color='red')
    plt.xlim((lb, ub))
    plt.text(0.5*ub, 0.6*np.max(hist), 'N = ' + str(len(counts)) +
            '\n\nn_exp = ' + str(np.round(exp_count, 2)) +
            '\n\nmean = ' + str(np.round(np.mean(counts), 2)))
    plt.text(0.5 * ub, 0.5 * np.max(hist), '\n\nmean_paired = ' + str(np.round(np.mean(counts_paired), 2)), color='blue')
    plt.text(0.5 * ub, 0.4 * np.max(hist), '\n\nmean_unpaired = ' + str(np.round(np.mean(counts_unpaired), 2)), color='red')
    plt.title(nt_x + ' --> ' + nt_y)
    plt.xlabel('mutation counts')
    plt.savefig(nt_x+nt_y+'.png')
    plt.show()

    if exp_count > 10:
        log_counts = np.log(df['actual_count'].values)
        log_counts_paired = log_counts[df['unpaired'].values == 0]
        log_counts_unpaired = log_counts[df['unpaired'].values == 1]

        [log_min, log_max] = [np.min(log_counts), np.max(log_counts)]

        [log_lb, log_ub] = [max(np.percentile(log_counts, 0.5), -1000), np.percentile(log_counts, 99.5)]

        log_n_bins = int(len(log_counts)/25)
        if exp_count < 30:
            log_n_bins = int(log_n_bins/4)

        if len(log_counts) < 900:
            log_n_bins = int(log_n_bins*3)

        log_hist, log_bin_edges = np.histogram(log_counts, bins=log_n_bins, range=(log_lb, log_ub))
        log_hist_paired, log_bin_edges_paired = np.histogram(log_counts_paired, bins=log_n_bins, range=(log_lb, log_ub))
        log_hist_unpaired, log_bin_edges_unpaired = np.histogram(log_counts_unpaired, bins=log_n_bins, range=(log_lb, log_ub))
        plt.text(log_lb, 0.8*np.max(log_hist), 'N = ' + str(len(log_counts)) +
               '\n\nmean = ' + str(np.round(np.mean(log_counts), 2)))
        plt.text(log_lb, 0.65 * np.max(log_hist), '\n\nmean_paired\n' + '= '+str(np.round(np.mean(log_counts_paired), 2)), color='blue')
        plt.text(log_lb, 0.5 * np.max(log_hist), '\n\nmean_unpaired\n' + '= '+str(np.round(np.mean(log_counts_unpaired), 2)), color='red')
        plt.bar(log_bin_edges[:-1], log_hist, width=(log_bin_edges[-1] - log_bin_edges[0]) / log_n_bins, alpha=0.7, align='edge', color='none',
                edgecolor='black')
        plt.bar(log_bin_edges_paired[:-1], log_hist_paired, width=(log_bin_edges[-1]-log_bin_edges[0])/log_n_bins,
                alpha=0.6, align='edge', color='blue')
        plt.bar(log_bin_edges_unpaired[:-1], log_hist_unpaired, width=(log_bin_edges[-1]-log_bin_edges[0])/log_n_bins,
                alpha=0.6, align='edge', color='red')

        plt.title(nt_x + ' --> ' + nt_y)
        plt.xlabel('log mutation counts')
        plt.savefig(nt_x+nt_y+'_log.png')
        plt.show()


    log_counts_trunc = log_counts[log_counts > pseudo_count]
    [lb, ub] = [np.percentile(log_counts_trunc, 0.3), np.percentile(log_counts_trunc, 99.9)]
    hist, bin_edges = np.histogram(log_counts, range=(np.log(pseudo_count), ub), bins=round(len(log_counts_trunc) / 20),
                                   density=True)
    # labels = gmm.predict(((bin_edges[1:] + bin_edges[:-1]) / 2).reshape(-1, 1))
    plt.figure(figsize=(8, 6))
    # plt.bar(bin_edges[:-1], hist, width=(bin_edges[-1] - bin_edges[0]) / round(len(log_counts) / 10), alpha=0.7,
    #         color=np.where(labels, 'green', 'blue'), align='edge')
    plt.bar(bin_edges[:-1], hist, width=(bin_edges[-1] - bin_edges[0]) / round(len(log_counts) / 20), alpha=0.7,
            align='edge')

    # Fit 1d Gaussian mixture model
    gmm = GaussianMixture(n_components=2, means_init=[[0.9 * log_exp_count], [1.1 * log_exp_count]],
                          n_init=5, random_state=1)
    gmm.fit(log_counts.reshape(-1, 1))

    # Plot histogram of log(counts)
    log_counts_trunc = log_counts[log_counts > pseudo_count]
    [lb, ub] = [np.percentile(log_counts_trunc, 0.3), np.percentile(log_counts_trunc, 99.9)]
    hist, bin_edges = np.histogram(log_counts, range=(np.log(pseudo_count), ub), bins=round(len(log_counts_trunc) / 20), density=True)
    #labels = gmm.predict(((bin_edges[1:] + bin_edges[:-1]) / 2).reshape(-1, 1))
    plt.figure(figsize=(8, 6))
    # plt.bar(bin_edges[:-1], hist, width=(bin_edges[-1] - bin_edges[0]) / round(len(log_counts) / 10), alpha=0.7,
    #         color=np.where(labels, 'green', 'blue'), align='edge')
    plt.bar(bin_edges[:-1], hist, width=(bin_edges[-1] - bin_edges[0]) / round(len(log_counts) / 20), alpha=0.7, align='edge')

    # Plot individual components of the Gaussian mixture model
    x = np.linspace(lb, ub, 1000)
    for i in range(2):
        mu = gmm.means_[i][0]
        sigma = np.sqrt(gmm.covariances_[i][0])
        weight = gmm.weights_[i]
        plt.plot(x, weight * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                 linewidth=3, color=['blue', 'green'][i])

    # Plot full pdf of the Gaussian mixture model
    mixture_pdf = np.zeros_like(x)
    for weight, mean, covariance in zip(gmm.weights_, gmm.means_, gmm.covariances_):
        mixture_pdf += weight * (1 / (np.sqrt(2 * np.pi * covariance[0, 0]))) * np.exp(
            -0.5 * ((x - mean[0]) ** 2 / covariance[0, 0]))
    plt.plot(x, mixture_pdf, linewidth=3, color='black')

    # Compute confusion matrix
    predicted = gmm.predict(log_counts.reshape(-1, 1))
    ground_truth = df['unpaired'].values
    conf_mat = np.round(confusion_matrix(ground_truth, predicted, normalize='all'), 3)

    # Compute paired and unpaired mean from experimental data
    paired_counts = df['actual_count'].values[np.where(df['unpaired'].values == 0)]
    unpaired_counts = df['actual_count'].values[np.where(df['unpaired'].values == 1)]
    paired_mean = np.log(np.mean(paired_counts))
    unpaired_mean = np.log(np.mean(unpaired_counts))

    # Show plot
    plt.text(0.95 * lb, 0.55 * np.max(hist),
             'GMM means:\n' + str(round(gmm.means_[0, 0], 2)) + '\n' + str(round(gmm.means_[1, 0], 2))
             + '\n\nexp. data means:\n' + str(round(paired_mean, 2)) + '\n' + str(round(unpaired_mean, 2))
             + '\n\nlog(n_exp): ' + str(round(log_exp_count, 2)))
    plt.text(0.75 * ub, 0.7 * np.max(hist), 'confusion matrix\n(0: paired, 1: unpaired):\n' + str(conf_mat) +
             '\n\nn = ' + str(len(log_counts)))
    plt.title('Gaussian mixture model for\n synonymous ' + nt_x + '->' + nt_y + ' mutations in 21J')
    plt.xlabel('log(counts)')
    plt.grid()
    plt.show()

    # Plot histogram for paired and unpaired sites
    log_counts_paired = log_counts[np.where(df['unpaired'].values == 0)]
    log_counts_unpaired = log_counts[np.where(df['unpaired'].values == 1)]
    w_paired = len(log_counts_paired) / (len(log_counts_paired) + len(log_counts_unpaired))
    w_unpaired = len(log_counts_unpaired) / (len(log_counts_paired) + len(log_counts_unpaired))
    hist_paired, bin_edges_paired = np.histogram(log_counts_paired, range=(lb, ub),
                                                 bins=round(len(log_counts) / 20), density=True)
    hist_unpaired, bin_edges_unpaired = np.histogram(log_counts_unpaired, range=(lb, ub),
                                                     bins=round(len(log_counts) / 20), density=True)
    plt.figure(figsize=(8, 6))
    plt.bar(bin_edges_paired[:-1], hist_paired*w_paired, width=(bin_edges[-1] - bin_edges[0]) /round(len(log_counts) / 20),
            alpha=0.6, color='red', align='edge')
    plt.bar(bin_edges_unpaired[:-1], hist_unpaired*w_unpaired, width=(bin_edges[-1] - bin_edges[0]) / round(len(log_counts) / 20),
            alpha=0.6, color='black', align='edge')
    plt.show()

    return 0


def gmm_fit_plot_2d(df, cell_type):

    x = df['actual_count'].values
    y = df['dms'].values
    y_min = np.min(y[y > 0])
    y = np.log10(df['dms'].values + y_min)

    x_paired = x[df['unpaired'].values == 0]
    y_paired = y[df['unpaired'].values == 0]
    x_unpaired = x[df['unpaired'].values == 1]
    y_unpaired = y[df['unpaired'].values == 1]
    [lb, ub] = [np.floor(np.percentile(x, 0.25)), np.ceil(np.percentile(x, 99))]

    # Plot DMS
    dms_paired, dms_edges_paired = np.histogram(y_paired, bins=50, range=(min(y), -1))
    dms_unpaired, dms_edges_unpaired = np.histogram(y_unpaired, bins=50, range=(min(y), -1))
    plt.bar(dms_edges_paired[:-1], dms_paired, width=(-1-min(y)) / 50, alpha=0.7, align='edge', color='blue',
            edgecolor='black')
    plt.bar(dms_edges_unpaired[:-1], dms_unpaired, width=(-1-min(y)) / 50, alpha=0.7, align='edge', color='red',
            edgecolor='black')
    plt.title(nt_x)
    plt.xlabel('log(DMS)')
    plt.savefig(nt_x + '_dms.png')
    plt.show()

    plt.figure(figsize=(8, 7))
    plt.scatter(x, y, color=np.where(df['unpaired'].values, 'red', 'blue'), s=15)
    plt.text(lb+5, np.log10(0.07), 'r_all = ' + str(round(pearsonr(x, y)[0], 3)) +
             '\nr_paired = ' + str(round(pearsonr(x_paired, y_paired)[0], 3)) +
             '\nr_unpaired =' + str(round(pearsonr(x_unpaired, y_unpaired)[0], 3)))
    plt.xlabel('mutation count')
    plt.ylabel('log(DMS)')
    plt.title(nt_x+' --> '+nt_y)
    plt.xlim((lb, ub))
    plt.ylim((min(y), -1))
    plt.grid()
    plt.savefig(nt_x + nt_y + '_2d.png')
    plt.show()

    x_log = np.log(df['actual_count'].values)
    #y = np.log10(df['dms'].values)

    [lb, ub] = [np.floor(np.percentile(x_log, 0.5)), np.ceil(np.percentile(x_log, 99.5))]

    x_paired = x_log[df['unpaired'].values == 0]
    y_paired = y[df['unpaired'].values == 0]
    x_unpaired = x_log[df['unpaired'].values == 1]
    y_unpaired = y[df['unpaired'].values == 1]

    plt.figure(figsize=(8, 7))
    plt.scatter(x_log, y, color=np.where(df['unpaired'].values, 'red', 'blue'), s=15)
    plt.text(lb+0.1, np.log10(0.07), 'r_all = ' + str(round(pearsonr(x_log, y)[0], 3)) +
             '\nr_paired = ' + str(round(pearsonr(x_paired, y_paired)[0], 3)) +
             '\nr_unpaired =' + str(round(pearsonr(x_unpaired, y_unpaired)[0], 3)))
    plt.xlabel('log(mutation count)')
    plt.ylabel('log(DMS)')
    plt.title(nt_x + ' --> ' + nt_y)
    plt.xlim((lb, ub))
    plt.ylim((min(y), -1))
    plt.grid()
    plt.savefig(nt_x + nt_y + '_2d_log.png')
    plt.show()


    Prepare data
    x = np.log(df['actual_count'] + 0.1).values
    y = df['dms'].values
    data = np.column_stack((x, y))

    Fit 2D Gaussian mixture model
    gmm_help = GaussianMixture(n_components=2, n_init=5, random_state=1)
    gmm_help.fit(data)
    gmm = GaussianMixture(n_components=2, means_init=[gmm_help.means_[np.argmin(gmm_help.means_[:, 0])],
                                                      gmm_help.means_[np.argmax(gmm_help.means_[:, 0])]], n_init=5)
    gmm.fit(data)

    # Calculate confusion matrix
    predicted = gmm.predict(data)
    ground_truth = df['unpaired'].values
    conf_mat = np.round(confusion_matrix(ground_truth, predicted, normalize='all'), 3)

    # Plot data classified by true labels
    x_paired = x[np.where(ground_truth == 0)]
    y_paired = y[np.where(ground_truth == 0)]
    x_unpaired = x[np.where(ground_truth == 1)]
    y_unpaired = y[np.where(ground_truth == 1)]

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, color=np.where(ground_truth, 'green', 'blue'), s=15)
    plt.text(2.5, 0.08, 'r_paired = ' + str(round(pearsonr(x_paired, y_paired)[0], 3)) +
             '\nr_unpaired =' + str(round(pearsonr(x_unpaired, y_unpaired)[0], 3)))
    plt.xlabel('log(count)')
    plt.ylabel('DMS')
    plt.title('data classified according\nto experimental data (' + cell_type + ')')
    plt.xlim((2, 7))
    plt.ylim((0, 0.1))
    plt.grid()
    plt.show()

    # Plot GMM and data classified by GMM
    x_range = np.linspace(2, 7, 100)
    y_range = np.linspace(0, 0.1, 100)
    X, Y = np.meshgrid(x_range, y_range)
    XY = np.column_stack((X.ravel(), Y.ravel()))

    densities = gmm.score_samples(XY)
    densities = densities.reshape(X.shape)

    x_paired = x[np.where(predicted == 0)]
    x_unpaired = x[np.where(predicted == 1)]
    y_paired = y[np.where(predicted == 0)]
    y_unpaired = y[np.where(predicted == 1)]

    plt.figure(figsize=(8, 8))
    plt.text(2.5, 0.08, 'r_paired = ' + str(round(pearsonr(x_paired, y_paired)[0], 3)) +
             '\nr_unpaired = ' + str(round(pearsonr(x_unpaired, y_unpaired)[0], 3)))
    plt.scatter(x, y, color=np.where(predicted, 'green', 'blue'), s=15)
    plt.contour(X, Y, densities, levels=np.logspace(-0.5, 2, 20, base=np.exp(1)))
    plt.xlabel('log(count)')
    plt.ylabel('DMS')
    plt.title('data classified according\nto 2D GMM (' + cell_type + ')')
    plt.text(2.5, 0.05,
             'confusion matrix\n(0: paired, 1: unpaired):\n' + str(conf_mat) + '\n\nn = ' + str(len(predicted)))
    plt.xlim((2, 7))
    plt.ylim((0, 0.1))
    plt.grid()
    plt.show()

    return 0 #gmm


if __name__ == '__main__':

    # Read file with mutation data
    df = pd.read_csv(
        '/Users/georgangehrn/Desktop/SARS2-mut-fitness-corr/human_data/counts/counts_all_21J_nonex_syn.csv')

    # Loop through all mutation types
    for nt_1 in ['A', 'C', 'G', 'T']:

        for nt_2 in ['A', 'C', 'G', 'T']:

            if nt_1 != nt_2:

                # Define mutation type X --> Y, cell type, and pseudo_count
                [nt_x, nt_y, cell_type, pseudo_count] = [nt_1, nt_2, 'Huh7', 0.5]

                # Only keep specified mutation type
                df_mut = df[df['nt_mutation'].str.match('^' + nt_x + '.*' + nt_y + '$')]

                # Take care of pseudocount and not null and expected count

    # Fit 1D Gaussian mixture model to log(counts) and plot results
    #gmm_1d = gmm_fit_plot_1d(df, nt_x, nt_y, exp_count, pseudo_count)

    # if nt_1 in ['A', 'C'] and exp_count > 10:
    #gmm_2d = gmm_fit_plot_2d(df, cell_type)



    actual_counts = df['actual_count'].values
    expected_counts = df['expected_count'].values
    n = len(actual_counts)
    fit_uncorr = np.log((actual_counts + 0.5) / (expected_counts + 0.5))
    [mean_uncorr, var_uncorr] = [np.mean(fit_uncorr), np.var(fit_uncorr)]

    # Using the mean values of un-/paired sited
    paired_counts = df['actual_count'].values[np.where(df['unpaired'] == 0)[0]]
    unpaired_counts = df['actual_count'].values[np.where(df['unpaired'].values == 1)[0]]
    paired_mean = np.exp(np.mean(np.log(paired_counts + 0.1)))  # What do the pseudocount and the exp(log()) do?
    unpaired_mean = np.exp(np.mean(np.log(unpaired_counts + 0.1)))

    fit_corr_1 = np.zeros(n)
    for i in range(n):
        if df['unpaired'].values[i] == 0:
            fit_corr_1[i] = np.log((actual_counts[i] + 0.5) / (paired_mean + 0.5))
        else:
            fit_corr_1[i] = np.log((actual_counts[i] + 0.5) / (unpaired_mean + 0.5))
    [mean_corr_1, var_corr_1] = [np.mean(fit_corr_1), np.var(fit_corr_1)]

    fit_corr_2 = np.zeros(n)
    fit_corr_5 = np.zeros(n)
    for i in range(n):
        if df['unpaired'].values[i] == 0:
            fit_corr_2[i] = np.log((actual_counts[i] + 0.5) / (np.exp(gmm_1d.means_[0, 0]) + 0.5))
            fit_corr_5[i] = np.log((actual_counts[i] + 0.5) / (np.exp(gmm_2d.means_[0, 0]) + 0.5))
        else:
            fit_corr_2[i] = np.log((actual_counts[i] + 0.5) / (np.exp(gmm_1d.means_[1, 0]) + 0.5))
            fit_corr_5[i] = np.log((actual_counts[i] + 0.5) / (np.exp(gmm_2d.means_[1, 0]) + 0.5))
    [mean_corr_2, var_corr_2] = [np.mean(fit_corr_2), np.var(fit_corr_2)]
    [mean_corr_5, var_corr_5] = [np.mean(fit_corr_5), np.var(fit_corr_5)]

    # Use GMM as classifier
    fit_corr_3 = np.zeros(n)
    fit_corr_6 = np.zeros(n)
    for i in range(n):
        if gmm_1d.predict(np.log(df['actual_count'].values[i].reshape(-1, 1) + 0.1)) == 0:
            fit_corr_3[i] = np.log((actual_counts[i] + 0.5) / (np.exp(gmm_1d.means_[0, 0]) + 0.5))
        else:
            fit_corr_3[i] = np.log((actual_counts[i] + 0.5) / (np.exp(gmm_1d.means_[1, 0]) + 0.5))

        if gmm_2d.predict([[np.log(df['actual_count'].values[i] + 0.1), df['dms'].values[i]]]) == 0:
            fit_corr_6[i] = np.log((actual_counts[i] + 0.5) / (np.exp(gmm_2d.means_[0, 0]) + 0.5))
        else:
            fit_corr_6[i] = np.log((actual_counts[i] + 0.5) / (np.exp(gmm_2d.means_[1, 0]) + 0.5))
    [mean_corr_3, var_corr_3] = [np.mean(fit_corr_3), np.var(fit_corr_3)]
    [mean_corr_6, var_corr_6] = [np.mean(fit_corr_6), np.var(fit_corr_6)]

    # Linear combination of GMM components
    fit_corr_4 = np.zeros(n)
    fit_corr_7 = np.zeros(n)
    for i in range(n):
        lin_comb_1d = gmm_1d.predict_proba(np.log(df['actual_count'].values[i] + 0.1).reshape(-1, 1))
        lin_comb_2d = gmm_2d.predict_proba([[np.log(df['actual_count'].values[i] + 0.1), df['dms'].values[i]]])
        lin_comb_1d = lin_comb_1d / (np.sum(lin_comb_1d))
        lin_comb_2d = lin_comb_2d / (np.sum(lin_comb_2d))
        mean_1d = lin_comb_1d[0, 0] * gmm_1d.means_[0, 0] + lin_comb_1d[0, 1] * gmm_1d.means_[1, 0]
        mean_2d = lin_comb_2d[0, 0] * gmm_2d.means_[0, 0] + lin_comb_2d[0, 1] * gmm_2d.means_[1, 0]
        fit_corr_4[i] = np.log((df['actual_count'].values[i] + 0.5) / (np.exp(mean_1d) + 0.5))
        fit_corr_7[i] = np.log((df['actual_count'].values[i] + 0.5) / (np.exp(mean_2d) + 0.5))
    [mean_corr_4, var_corr_4] = [np.mean(fit_corr_4), np.var(fit_corr_4)]
    [mean_corr_7, var_corr_7] = [np.mean(fit_corr_7), np.var(fit_corr_7)]

    # Create map for histogram texts (improve this)
    map_fit = {1: fit_corr_1, 2: fit_corr_2, 3: fit_corr_3, 4: fit_corr_4,
               5: fit_corr_5, 6: fit_corr_6, 7: fit_corr_7}
    map_mean = {1: mean_corr_1, 2: mean_corr_2, 3: mean_corr_3, 4: mean_corr_4,
                5: mean_corr_5, 6: mean_corr_6, 7: mean_corr_7}
    map_var = {1: var_corr_1, 2: var_corr_2, 3: var_corr_3, 4: var_corr_4,
               5: var_corr_5, 6: var_corr_6, 7: var_corr_7}
    map_title = {1: 'using mean of un-/paired sites', 2: '1D GMM, classifier: exp',
                 5: '2D GMM, classifier: exp', 3: '1D GMM, classifier: GMM',
                 6: '2D GMM, classifier: GMM', 4: '1D GMM, classifier: lin. comb.',
                 7: '2D GMM, classifier: lin. comb.'}

    # Create histograms for corrected fitness
    for i in range(1, 8):
        plt.figure(figsize=(8, 6))

        plt.hist(fit_uncorr, bins=60, alpha=0.7, color='red', range=(-3 * var_uncorr, 3 * var_uncorr), density=True,
                 label='uncorrected')
        plt.text(-2.5,
                 0.8 * max(np.histogram(map_fit[i], bins=60, density=True, range=(-3 * var_uncorr, 3 * var_uncorr))[0]),
                 'mean = ' + str(round(mean_uncorr, 3)), fontsize=12, color='red')
        plt.text(-2.5,
                 0.7 * max(np.histogram(map_fit[i], bins=60, density=True, range=(-3 * var_uncorr, 3 * var_uncorr))[0]),
                 'var = ' + str(round(var_uncorr, 3)), fontsize=12, color='red')

        plt.hist(map_fit[i], bins=60, alpha=0.7, color='black', range=(-3 * var_uncorr, 3 * var_uncorr), density=True,
                 label='corrected')
        plt.text(1.5,
                 0.8 * max(np.histogram(map_fit[i], bins=60, density=True, range=(-3 * var_uncorr, 3 * var_uncorr))[0]),
                 'mean = ' + str(round(map_mean[i], 3)), fontsize=12, color='black')
        plt.text(1.5,
                 0.7 * max(np.histogram(map_fit[i], bins=60, density=True, range=(-3 * var_uncorr, 3 * var_uncorr))[0]),
                 'var = ' + str(round(map_var[i], 3)), fontsize=12, color='black')

        plt.xlabel('nt fitness effect')
        plt.title(str(i) + ')   ' + map_title[i])
        plt.legend()
        plt.show()
