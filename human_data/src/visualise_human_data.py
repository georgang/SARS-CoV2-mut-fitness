import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

plot_map = {'AC': [0, 0], 'CA': [0, 1], 'GA': [0, 2], 'TA': [0, 3],
            'AG': [1, 0], 'CG': [1, 1], 'GC': [1, 2], 'TC': [1, 3],
            'AT': [2, 0], 'CT': [2, 1], 'GT': [2, 2], 'TG': [2, 3]
            }

if __name__ == '__main__':

    '''
        This script visualises the mean mutation counts for every mutation type.
    '''

    # Read file with mutation data
    df_mut = pd.read_csv('../counts/counts_all_21K_nonex_syn.csv')

    # Prepare dictionary for means
    mean_abs_counts = {}
    mean_log_counts = {}
    pseudo_count = 0.5

    # Loop through all 12 mutation types
    for nt_1 in ['A', 'C', 'G', 'T']:

        for nt_2 in ['A', 'C', 'G', 'T']:

            if nt_1 != nt_2:

                # Only keep one mutation type
                df = df_mut[df_mut['nt_mutation'].str.match('^' + nt_1 + '.*' + nt_2 + '$')]

                # Extract mutation counts from dataframe
                counts = df['actual_count'].values
                counts_paired = counts[df['unpaired'].values == 0]
                counts_unpaired = counts[df['unpaired'].values == 1]
                mean_abs_counts[nt_1+nt_2] = [np.mean(counts_paired), np.mean(counts_unpaired)]
                mean_log_counts[nt_1+nt_2] = [np.log(np.mean(counts_paired) + pseudo_count),
                                              np.log(np.mean(counts_unpaired) + pseudo_count)]

    # Plot mean absolute/log counts for all mutation types in one plot
    for i, mean_type in enumerate([mean_abs_counts, mean_log_counts]):
        keys = list(mean_type.keys())
        values = list(mean_type.values())
        paired_values = [val[0] for val in values]
        unpaired_values = [val[1] for val in values]
        bar_width = 0.4

        r1 = range(len(keys))
        r2 = [x + bar_width for x in r1]

        plt.figure(dpi=200)

        plt.bar(r1, paired_values, color='b', width=bar_width, label='paired', alpha=0.6)
        plt.bar(r2, unpaired_values, color='r', width=bar_width, label='unpaired', alpha=0.6)

        plt.xlabel('mutation type', fontsize=12)
        if i == 0:
            plt.ylabel('mean absolute mutation count', fontsize=12)
        else:
            plt.ylabel('mean log mutation count', fontsize=12)
            plt.axhline(y=np.log(pseudo_count), linestyle='--', color='black', label='ln('+str(pseudo_count)+')')
            plt.axhline(y=0, linestyle='-', color='black')
        plt.xticks([r + bar_width / 2 for r in range(len(keys))], keys)

        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)

        plt.legend(fontsize=10)

        plt.tight_layout()
        plt.show()

    # Plot relative amount of mutations
    rel_values = np.vstack(list(mean_abs_counts.values()))
    rel_values = np.hstack([rel_values, np.sum(rel_values, axis=1).reshape(-1, 1)])
    rel_values = rel_values / np.sum(rel_values[:, 2], axis=0)

    plt.figure(dpi=200)

    x_ticks = np.arange(len(keys))
    plt.xticks(x_ticks, keys)

    plt.bar(x_ticks, rel_values[:, 0], color='b', label='paired', alpha=0.6)
    plt.bar(x_ticks, rel_values[:, 1], bottom=rel_values[:, 0], color='r', label='unpaired', alpha=0.6)
    plt.bar(x_ticks, rel_values[:, 2], color='none', edgecolor='black', label='total')

    plt.xlabel('mutation type', fontsize=12)
    plt.ylabel('relative amount', fontsize=12)

    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)

    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.show()
