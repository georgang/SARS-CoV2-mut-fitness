import numpy as np
from helper import mut_types, load_mut_counts
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})


def plot_relative_mut_counts(mean_counts, title):
    # mean_counts is a dictionary with the mutation types as keys
    rel_values = np.vstack(list(mean_counts.values()))
    rel_values = np.hstack([rel_values, np.sum(rel_values, axis=1).reshape(-1, 1)])
    rel_values = rel_values / np.sum(rel_values[:, 2], axis=0)

    plt.figure()

    x_ticks = np.arange(12)
    plt.xticks(x_ticks, mut_types)

    width = 0.3
    plt.bar(x_ticks + width, rel_values[:, 0], width=width, color='b', label='paired', alpha=0.6)
    plt.bar(x_ticks, rel_values[:, 1], width=width, color='r', label='unpaired', alpha=0.6)
    plt.bar(x_ticks - width, rel_values[:, 2], width=width, color='black', label='total', alpha=0.8)

    plt.xlabel('mutation type', fontsize=12)
    plt.ylabel('relative amount', fontsize=12)

    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)

    plt.legend(fontsize=10)

    plt.ylim((0, 0.55))

    plt.title(title)

    plt.tight_layout()
    plt.show()


def plot_absolute_counts(mean_counts, mean_log_counts, title):

    # Loop over both mean types
    for i, mean_type in enumerate([mean_counts, mean_log_counts]):

        keys = list(mean_type.keys())
        values = list(mean_type.values())

        paired_values = [val[0] for val in values]
        unpaired_values = [val[1] for val in values]

        bar_width = 0.4

        r1 = range(len(keys))
        r2 = [x + bar_width for x in r1]

        plt.figure()

        plt.bar(r1, paired_values, color='b', width=bar_width, label='paired', alpha=0.6)
        plt.bar(r2, unpaired_values, color='r', width=bar_width, label='unpaired', alpha=0.6)

        plt.xlabel('mutation type', fontsize=12)
        if i == 0:
            plt.ylabel('mean absolute mutation count', fontsize=12)
        else:
            plt.ylabel('mean log mutation count', fontsize=12)
            plt.axhline(y=np.log(0.5), linestyle='--', color='black', label='ln(' + str(0.5) + ')')
            plt.axhline(y=0, linestyle='-', color='black')
        plt.xticks([r + bar_width / 2 for r in range(len(keys))], keys)

        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)

        plt.legend(fontsize=10)

        plt.title(title)
        plt.tight_layout()
        plt.show()


def get_mean_counts(df_mut, pseudo_count):

    # Prepare dictionaries for mean (log) counts
    mean_counts = {}
    mean_log_counts = {}

    # Loop through all 12 mutation types
    for mut_type in mut_types:

        # Only keep one mutation type
        df = df_mut[df_mut['nt_mutation'].str.match('^' + mut_type[0] + '.*' + mut_type[1] + '$')]

        # Extract mutation counts for this mutation type from dataframe
        counts = df['actual_count'].values

        counts_paired = counts[df['unpaired'].values == 0]
        counts_unpaired = counts[df['unpaired'].values == 1]

        mean_counts[mut_type] = [np.mean(counts_paired), np.mean(counts_unpaired)]

        mean_log_counts[mut_type] = [np.log(np.mean(counts_paired) + pseudo_count),
                                     np.log(np.mean(counts_unpaired) + pseudo_count)]

    return mean_counts, mean_log_counts


if __name__ == '__main__':
    '''This script visualises the mean mutation counts for every mutation type. I mainly used this for comparing 21J 
    and 21K.'''

    CLADE = '21J'

    # Read file with mutation data
    df_all = load_mut_counts(CLADE)

    # Get mean (log) count for every mutation type
    mean_counts, mean_log_counts = get_mean_counts(df_all, pseudo_count=0.5)

    # Plot mean (log) counts
    # TODO: Make sure that the following two functions are correct
    # TODO: Find a better solution for keeping the same y_lims when comparing clades
    plot_absolute_counts(mean_counts=mean_counts, mean_log_counts=mean_log_counts, title=CLADE)

    # Plot relative amount of mutations
    plot_relative_mut_counts(mean_counts=mean_counts, title=CLADE)
