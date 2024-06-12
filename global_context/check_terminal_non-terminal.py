import numpy as np
from helper import load_mut_counts, mut_types, plot_map
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 12})


# Genome length
REF_LENGTH = 29903

# Gene boundaries
genes = []
with open('/Users/georgangehrn/Desktop/SARS2-mut-fitness-corr/human_data/genes.txt', 'r') as file:
    for line in file:
        items = line.strip().split(',')
        genes.append((int(items[0]), int(items[1]), items[2]))


if __name__ == '__main__':

    data = load_mut_counts('21J')

    PERCENTILE = 95  # Put to None if no clipping is desired

    # Loop over different averaging window sizes
    for window_len in [1001, 2001, 3001]:

        fig, axes = plt.subplots(3, 4, figsize=(16, 9))

        for mut_type in mut_types:

            ax = axes[plot_map[mut_type]]

            df = data[data['nt_mutation'].str.match('^' + mut_type[0] + '.*' + mut_type[1] + '$')]

            for state in ['', 'non_']:

                sites = df['nt_site'].values
                counts = df['count_'+state+'terminal'].values

                if PERCENTILE is not None:
                    clip_value = np.percentile(counts, PERCENTILE)
                    counts = np.clip(counts, a_min=None, a_max=clip_value)

                overall_avg = np.average(counts)

                half_wndw = (window_len - 1) / 2

                roll_avg = np.zeros(int(REF_LENGTH - 2 * half_wndw))
                new_sites = np.arange(half_wndw + 1, REF_LENGTH - half_wndw + 1)

                for i in range(len(new_sites)):
                    lower_bound = new_sites[i] - half_wndw
                    upper_bound = new_sites[i] + half_wndw
                    mask = (sites >= lower_bound) & (sites <= upper_bound)
                    roll_avg[i] = np.average(counts[mask])

                ax.plot(new_sites, roll_avg / overall_avg, label=state+'terminal', color={'': 'green', 'non_': 'orange'}[state])
                if plot_map[mut_type][0] == 2:
                    ax.set_xlabel('nucleotide position')
                if plot_map[mut_type][1] == 0:
                    ax.set_ylabel('rolling/overall mean count')
                ax.legend()
                ax.set_xticks([1] + [genes[i][0] for i in range(len(genes))] + [REF_LENGTH])
                ax.set_xticklabels(['', 'ORF1a', 'ORF1b', 'S', '(...)'] + [''] * (len(genes) - 4) + [''],
                                   ha='left')
                ax.grid(axis='x', which='both', linestyle='-')
                ax.axhline(1, color='black', linestyle='-')
                ax.set_title(mut_type[0] + r'$\rightarrow$' + mut_type[0])

        plt.suptitle(f'data: synonymous, non-excluded mutations in 21J, counts clipped at {PERCENTILE if PERCENTILE is not None else 100}-th percentile, sliding window: {window_len}')
        plt.tight_layout()
        plt.savefig(f'results/check_terminal/check_terminal_{window_len}.png')
        plt.show()
