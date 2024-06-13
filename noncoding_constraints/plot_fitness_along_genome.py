import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from helper import load_mut_counts, gene_boundaries
from general_linear_models.fit_general_linear_models import add_predictions

plt.rcParams.update({'font.size': 12})

REF_LENGTH = 29903


def compute_rolling_average(window_len, sites, values):

    # Get size of half the window
    half_wndw = (window_len - 1) / 2

    # Prepare arrays for nt sites and rolling averages that exclude edges of the genome
    new_sites = np.arange(half_wndw + 1, REF_LENGTH - half_wndw + 1)
    roll_avg = np.zeros(int(REF_LENGTH - 2 * half_wndw))
    n_of_datapoints = np.zeros(int(REF_LENGTH - 2 * half_wndw))

    # Compute the rolling average for every nt site
    for i, site in enumerate(new_sites):

        # Get all sites in the dataset that are within the current window boundaries
        lower_bound = site - half_wndw
        upper_bound = site + half_wndw
        mask = (sites >= lower_bound) & (sites <= upper_bound)

        roll_avg[i] = np.mean(values[mask])
        n_of_datapoints[i] = len(values[mask])

    return new_sites, roll_avg, n_of_datapoints


def plot_rolling_avg(new_sites, roll_avg_uncorr, roll_avg_corr, nt_lims=(1, REF_LENGTH)):

    # Plot fitness along genome
    plt.figure(figsize=(16, 4), dpi=200)
    plt.plot(new_sites, roll_avg_uncorr, label='uncorrected', color='black', alpha=0.7)
    plt.plot(new_sites, roll_avg_corr, label='corrected', color='green')
    plt.axhline(0, color='black')
    plt.ylim((-4.2, 1.2))
    plt.ylabel('fitness effect')
    plt.xlabel('nucleotide position')
    plt.title(f"sliding window: {WINDOW_LEN}")
    plt.legend()

    # Add gene boundaries to plot
    colors = cm.get_cmap('tab20', len(gene_boundaries))
    for i, (start, end, name) in enumerate(gene_boundaries):
        if start > nt_lims[0]:
            plt.barh(y=-3 + (i % 2) * 0.2, width=end - start, left=start, height=0.2, color=colors(i), alpha=0.7)
            plt.text(start, -3 + (i % 2) * 0.2, name, ha='left', va='center', fontsize=10)
        elif end > nt_lims[0]:
            plt.barh(y=-3 + (i % 2) * 0.2, width=end - nt_lims[0], left=nt_lims[0], height=0.2, color=colors(i), alpha=0.7)
            plt.text(nt_lims[0], -3 + (i % 2) * 0.2, name, ha='left', va='center', fontsize=10)

    if nt_lims == (1, REF_LENGTH):
        plt.text(0.5, 0.1, f"$\mu$ = {np.nanmean(roll_avg_uncorr):.2f}, var: {np.nanvar(roll_avg_uncorr):.2f}", transform=plt.gca().transAxes, color='black', ha='center', fontsize=10)
        plt.text(0.5, 0.05, f"$\mu$ = {np.nanmean(roll_avg_corr):.2f}, var: {np.nanvar(roll_avg_corr):.2f}", transform=plt.gca().transAxes, color='green', ha='center', fontsize=10)

    plt.title(f"{CLADE}, sliding window length: {WINDOW_LEN}")
    plt.xlim(nt_lims)
    plt.tight_layout()
    plt.show()


def plot_n_of_datapoints(new_sites, n_of_datapoints):

    plt.figure(figsize=(16, 4))
    plt.plot(new_sites, n_of_datapoints)
    plt.xlim((new_sites[0], new_sites[-1]))
    colors = cm.get_cmap('tab20', len(gene_boundaries))
    for i, (start, end, name) in enumerate(gene_boundaries):
        plt.barh(y=10 + (i % 2) * 2, width=end - start, left=start, height=2, color=colors(i), alpha=0.7)
        plt.text(start, 10 + (i % 2) * 2, name, ha='left', va='center', fontsize=10)
    plt.xlabel('nucleotide position')
    plt.ylabel('# of mutations used for rolling average')
    plt.title(f"{CLADE}, sliding window length: {WINDOW_LEN}")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(16, 4))
    plt.plot(new_sites, n_of_datapoints)
    plt.xlim((new_sites[0], new_sites[-1]))
    colors = cm.get_cmap('tab20', len(gene_boundaries))
    for i, (start, end, name) in enumerate(gene_boundaries):
        if start > 25000:
            plt.barh(y=10 + (i % 2) * 2, width=end - start, left=start, height=2, color=colors(i), alpha=0.7)
            plt.text(start, 10 + (i % 2) * 2, name, ha='left', va='center', fontsize=10)
    plt.xlim((25000, REF_LENGTH))
    plt.xlabel('nucleotide position')
    plt.ylabel('# of mutations used for rolling average')
    plt.title(f"{CLADE}, sliding window length: {WINDOW_LEN}")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    CLADE = '21J'
    WINDOW_LEN = 25

    # Load mutation counts data
    df = load_mut_counts(clade=CLADE, include_noncoding=True)

    # Add predicted ln(counts + 0.5) to dataframe
    df = add_predictions(df.copy(), clade=CLADE)

    # Extract counts
    sites = df['nt_site'].values
    obs_log_counts = np.log(df['actual_count'].values + 0.5)
    exp_log_counts = np.log(df['expected_count'].values + 0.5)
    pred_log_counts = df['pred_log_count'].values

    # Compute uncorrected and corrected fitness
    fitness_uncorr = obs_log_counts - exp_log_counts
    fitness_corr = obs_log_counts - pred_log_counts

    # Compute rolling averages of fitness
    new_sites, roll_avg_uncorr, n_of_datapoints = compute_rolling_average(WINDOW_LEN, sites, fitness_uncorr)
    _, roll_avg_corr, _ = compute_rolling_average(WINDOW_LEN, sites, fitness_corr)

    # Plot fitness across genome and across structural/accessory proteins
    plot_rolling_avg(new_sites, roll_avg_uncorr, roll_avg_corr)
    plot_rolling_avg(new_sites, roll_avg_uncorr, roll_avg_corr, nt_lims=(25000, REF_LENGTH))

    # Plot number of datapoints in sliding windows
    plot_n_of_datapoints(new_sites, n_of_datapoints)
