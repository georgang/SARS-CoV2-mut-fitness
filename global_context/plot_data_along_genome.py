import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from matplotlib import pyplot as plt
from helper import mut_types, plot_map, load_mut_counts
from general_linear_models.fit_general_linear_models import GeneralLinearModel

import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

plt.rcParams.update({'font.size': 12})


# Length of genome
REF_LENGTH = 29903

# Load gene boundaries
genes = []
with open('/Users/georgangehrn/Desktop/SARS2-mut-fitness-corr/human_data/genes.txt', 'r') as file:
    for line in file:
        items = line.strip().split(',')
        genes.append((int(items[0]), int(items[1]), items[2]))


def predict_log_counts(model_type, dataframe):

    glm = GeneralLinearModel(type=model_type, test_data=dataframe)

    # Load saved parameter
    glm.W = pd.read_csv(f"/Users/georgangehrn/Desktop/SARS2-mut-fitness-corr/general_linear_models/results/{CLADE}/{model_type}/learned_params.csv")

    # Predict log(counts + 0.5)
    predicted_log_counts = glm.predict_log_counts()

    return predicted_log_counts


def analyze_correlation(traces):

    # Calculate correlations between traces of different mutation types
    correlations = {}
    correlation_matrix = np.zeros((len(traces), len(traces)))
    for i, mut_type_1 in enumerate(mut_types):
        for j, mut_type_2 in enumerate(mut_types):
            if j > i:
                trace_1 = traces[mut_type_1]
                trace_2 = traces[mut_type_2]
                corr_value = stats.pearsonr(trace_1, trace_2)[0]
                correlations[mut_type_1 + "_" + mut_type_2] = corr_value
                correlation_matrix[i, j] = corr_value
                correlation_matrix[j, i] = corr_value
    np.fill_diagonal(correlation_matrix, 1)

    # Sort correlations
    sorted_correlations = sorted(correlations.items(), key=lambda item: item[1])

    # Extract the 12 highest correlations
    highest_12 = sorted_correlations[-12:]

    # Plot the 12 highest correlated pairs
    plot_scatter_plots(highest_12, f'12 highest correlated mutation types, model: {PREDICTION[:-4]}', correlations)

    # Plot the correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, xticklabels=mut_types, yticklabels=mut_types, annot=True, cmap='coolwarm', vmin=-1,
                vmax=1)
    plt.title(f'correlation matrix, model: {PREDICTION[:-4]}')
    plt.show()

    # Calculate distance matrix from correlation matrix
    dist_matrix = np.sqrt(2 * (1 - correlation_matrix))

    # Perform hierarchical clustering
    # TODO: Check how exactly this works and improve the dendrogram plot
    linked = sch.linkage(squareform(dist_matrix), 'complete')

    # Plot dendrogram
    plt.figure(figsize=(10, 7))
    sch.dendrogram(linked, labels=mut_types)
    plt.show()


def plot_scatter_plots(pairs, title, correlations):

    fig, axs = plt.subplots(3, 4, figsize=(16, 9))
    fig.suptitle(title)
    axs = axs.flatten()

    for idx, (pair_name, _) in enumerate(pairs):

        mut_type_1, mut_type_2 = pair_name.split("_")

        trace_1 = traces[mut_type_1]
        trace_2 = traces[mut_type_2]

        # Color points in scatter plot according to which part of the genome they belong to
        # TODO: Check this assignment again
        colors = ['blue'] * (genes[0][1] - int(new_sites[0])) + ['red'] * (genes[1][1] - genes[1][0])
        colors = colors + ['green'] * (26903 - len(colors))

        axs[idx].scatter(trace_1, trace_2, c=colors, s=2)

        axs[idx].text(0.1, 0.9, f"r = {correlations[pair_name]:.2f}", transform=axs[idx].transAxes, fontsize=10)
        axs[idx].text(0.1, 0.8, "ORF1a", transform=axs[idx].transAxes, fontsize=10, color='blue')
        axs[idx].text(0.1, 0.7, "ORF1b", transform=axs[idx].transAxes, fontsize=10, color='red')
        axs[idx].text(0.1, 0.6, "stru+acc", transform=axs[idx].transAxes, fontsize=10, color='green')

        axs[idx].set_xlabel(f'fitness estimate {mut_type_1}')
        axs[idx].set_ylabel(f'fitness estimate {mut_type_2}')

        # Calculate and plot linear fit
        slope, intercept = np.polyfit(trace_1, trace_2, 1)
        fit_line = np.poly1d([slope, intercept])
        axs[idx].plot(trace_1, fit_line(trace_1), color='black')

        axs[idx].grid()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # TODO: I should actually always add a log file to every plot with the settings during the data load.
    # TODO: Maybe organise this in a bit more readable way

    CLADE = '21J'

    PERCENTILE = 95  # Put to None if no clipping is desired
    SHOW_FOLD_CHANGE = False  # Make this False if PREDICTION is not None
    PREDICTION = 'l_r_log'  # Make this None if no comparison to model. Add "_log" if log counts should be compared
    MAKE_Y_LIMS_EQUAL = True  # TODO: Currently, the limits are set manually
    ANALYZE_CORRELATION = True
    SAVE_FIG = False

    FILENAME = f'{CLADE}_clipped-at-{PERCENTILE}{"_fold-change" if SHOW_FOLD_CHANGE else ""}{"_" + PREDICTION if PREDICTION is not None else ""}'

    # Load counts for non-excluded synonymous mutations in the selected clade
    data = load_mut_counts(clade=CLADE)

    # Prepare one figure for all 12 mutation types
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))

    # Keep track of lowest and highest y value in order to equalize the y limits of the subplots in the final figure
    y_min, y_max = 1000, -1000

    # Prepare dictionary to store traces of all mutation types
    traces = {}

    # Loop over all mutation types
    for mut_type in mut_types:

        # Extract ancestral (nt1) and mutant (nt2) nucleotide type
        nt1, nt2 = mut_type[0], mut_type[1]

        # Select corresponding subplot
        ax = axes[plot_map[nt1 + nt2]]

        # Select corresponding mutation count data
        df = data[data['nt_mutation'].str.match('^' + nt1 + '.*' + nt2 + '$')]

        # Extract nucleotide sites and mutation counts present in the dataset
        sites = df['nt_site'].values
        counts = df['actual_count'].values
        N_of_sites = len(sites)

        # Clip outliers
        if PERCENTILE is not None:
            clip_value = np.percentile(counts, PERCENTILE)
            counts = np.clip(counts, a_min=None, a_max=clip_value)

        # Get average mutation count over all sites present in the data
        overall_avg = np.average(counts)

        # Predict counts using a trained general linear model
        if PREDICTION is not None:
            if PREDICTION[-3:] == 'log':
                predicted_log_counts = predict_log_counts(PREDICTION[:-4], df)
            else:
                predicted_counts = np.exp(predict_log_counts(PREDICTION, df)) - 0.5

        # Loop over different window sizes
        for window_len in [1001, 2001, 3001]:

            # Get size of half the window
            half_wndw = (window_len - 1) / 2

            # Prepare arrays for nt sites and rolling averages that exclude edges of the genome
            new_sites = np.arange(half_wndw + 1, REF_LENGTH - half_wndw + 1)
            roll_avg = np.zeros(int(REF_LENGTH - 2 * half_wndw))

            # Compute the rolling average for every nt site
            for i, site in enumerate(new_sites):

                # Get all sites in the dataset that are within the current window boundaries
                lower_bound = site - half_wndw
                upper_bound = site + half_wndw
                mask = (sites >= lower_bound) & (sites <= upper_bound)

                # Calculate the desired quantity for these sites
                if SHOW_FOLD_CHANGE:
                    roll_avg[i] = np.mean(counts[mask]) / overall_avg
                elif PREDICTION is not None:
                    if PREDICTION[-3:] == 'log':
                        roll_avg[i] = np.mean(np.log(counts[mask] + 0.5) - predicted_log_counts[mask])
                    else:
                        roll_avg[i] = np.mean(counts[mask] / predicted_counts[mask])
                else:
                    roll_avg[i] = np.mean(counts[mask])

            traces[mut_type] = roll_avg

            # Update minimal/maximal y-values
            y_min, y_max = min(y_min, np.min(roll_avg)), max(y_max, np.max(roll_avg))

            # Add trace to plot
            ax.plot(new_sites, roll_avg, label=f'{window_len} bp')

        # Add number of available sites to plot
        ax.text(0.1, 0.05, f"N = {N_of_sites}", transform=ax.transAxes, fontsize=8)

        # Set x-label
        if plot_map[nt1 + nt2][0] == 2:
            ax.set_xlabel('nucleotide position')

        # Set y-label and add horizontal lines for readability
        if plot_map[nt1 + nt2][1] == 0:
            if SHOW_FOLD_CHANGE:
                ax.set_ylabel(f'rolling/overall mean count', fontsize=10)
            elif PREDICTION:
                if PREDICTION[-3:] == 'log':
                    ax.set_ylabel(f'rolling mean of\nobs. - pred. log(count+0.5)', fontsize=10)
                else:
                    ax.set_ylabel(f'rolling mean of\nobs. / pred. count', fontsize=10)
            else:
                ax.set_ylabel(f'rolling mean count', fontsize=10)

        # Add horizontal lines for readability
        if SHOW_FOLD_CHANGE:
            ax.axhline(1, color='black', linestyle='--')
        elif PREDICTION:
            if PREDICTION[-3:] == 'log':
                ax.axhline(0, color='black', linestyle='--')
            else:
                ax.axhline(1, color='black', linestyle='--')
            ax.axhline(np.average(roll_avg), color='green', linestyle='--')
        else:
            ax.axhline(overall_avg, color='black', linestyle='--')

        # Set gene boundaries as x-ticks
        ax.set_xticks([1] + [genes[i][0] for i in range(len(genes))] + [REF_LENGTH])
        ax.set_xticklabels(['', 'ORF1a', 'ORF1b', 'S', '(...)'] + [''] * (len(genes) - 4) + [''], ha='left', fontsize=10)
        ax.grid(axis='x')

        # Add legend and title to subplot
        ax.legend(fontsize=10)
        ax.set_title(nt1 + r'$\rightarrow$' + nt2)

    # Equalize y-limits across subplots
    if MAKE_Y_LIMS_EQUAL:
        for mut_type in mut_types:
            ax = axes[plot_map[mut_type]]
            # ax.set_ylim(math.floor(10 * y_min) / 10, math.ceil(10 * y_max) / 10)
            ax.set_ylim(-1, 1)

    # Add overall title
    model_suffix = PREDICTION[:-4] if PREDICTION[-3:] == "log" else PREDICTION
    plt.suptitle(f'data: synonymous, non-excluded mutations in {CLADE}, counts clipped at {PERCENTILE if PERCENTILE is not None else 100}-th percentile{", gen.lin. model: " + model_suffix if PREDICTION is not None else ""} ')

    # Save and show full figure
    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig(f"results/{FILENAME}.png")
    plt.show()

    if ANALYZE_CORRELATION:
        # TODO: So far, this analysis only shows what is expected anyways
        analyze_correlation(traces)
