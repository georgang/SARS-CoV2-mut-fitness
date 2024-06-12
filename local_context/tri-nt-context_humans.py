import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 9})

letters = ['A', 'C', 'G', 'T']

plot_map = {'AC': [0, 0], 'CA': [0, 1], 'GA': [0, 2], 'TA': [0, 3],
            'AG': [1, 0], 'CG': [1, 1], 'GC': [1, 2], 'TC': [1, 3],
            'AT': [2, 0], 'CT': [2, 1], 'GT': [2, 2], 'TG': [2, 3]
            }

row_keys = ['AAA_C_p', 'AAC_C_p', 'AAG_C_p', 'AAT_C_p', 'CAA_C_p', 'CAC_C_p', 'CAG_C_p', 'CAT_C_p', 'GAA_C_p', 'GAC_C_p', 'GAG_C_p', 'GAT_C_p', 'TAA_C_p', 'TAC_C_p', 'TAG_C_p', 'TAT_C_p', 'AAA_C_up', 'AAC_C_up', 'AAG_C_up', 'AAT_C_up', 'CAA_C_up', 'CAC_C_up', 'CAG_C_up', 'CAT_C_up', 'GAA_C_up', 'GAC_C_up', 'GAG_C_up', 'GAT_C_up', 'TAA_C_up', 'TAC_C_up', 'TAG_C_up', 'TAT_C_up', 'AAA_G_p', 'AAC_G_p', 'AAG_G_p', 'AAT_G_p', 'CAA_G_p', 'CAC_G_p', 'CAG_G_p', 'CAT_G_p', 'GAA_G_p', 'GAC_G_p', 'GAG_G_p', 'GAT_G_p', 'TAA_G_p', 'TAC_G_p', 'TAG_G_p', 'TAT_G_p', 'AAA_G_up', 'AAC_G_up', 'AAG_G_up', 'AAT_G_up', 'CAA_G_up', 'CAC_G_up', 'CAG_G_up', 'CAT_G_up', 'GAA_G_up', 'GAC_G_up', 'GAG_G_up', 'GAT_G_up', 'TAA_G_up', 'TAC_G_up', 'TAG_G_up', 'TAT_G_up', 'AAA_T_p', 'AAC_T_p', 'AAG_T_p', 'AAT_T_p', 'CAA_T_p', 'CAC_T_p', 'CAG_T_p', 'CAT_T_p', 'GAA_T_p', 'GAC_T_p', 'GAG_T_p', 'GAT_T_p', 'TAA_T_p', 'TAC_T_p', 'TAG_T_p', 'TAT_T_p', 'AAA_T_up', 'AAC_T_up', 'AAG_T_up', 'AAT_T_up', 'CAA_T_up', 'CAC_T_up', 'CAG_T_up', 'CAT_T_up', 'GAA_T_up', 'GAC_T_up', 'GAG_T_up', 'GAT_T_up', 'TAA_T_up', 'TAC_T_up', 'TAG_T_up', 'TAT_T_up', 'ACA_A_p', 'ACC_A_p', 'ACG_A_p', 'ACT_A_p', 'CCA_A_p', 'CCC_A_p', 'CCG_A_p', 'CCT_A_p', 'GCA_A_p', 'GCC_A_p', 'GCG_A_p', 'GCT_A_p', 'TCA_A_p', 'TCC_A_p', 'TCG_A_p', 'TCT_A_p', 'ACA_A_up', 'ACC_A_up', 'ACG_A_up', 'ACT_A_up', 'CCA_A_up', 'CCC_A_up', 'CCG_A_up', 'CCT_A_up', 'GCA_A_up', 'GCC_A_up', 'GCG_A_up', 'GCT_A_up', 'TCA_A_up', 'TCC_A_up', 'TCG_A_up', 'TCT_A_up', 'ACA_G_p', 'ACC_G_p', 'ACG_G_p', 'ACT_G_p', 'CCA_G_p', 'CCC_G_p', 'CCG_G_p', 'CCT_G_p', 'GCA_G_p', 'GCC_G_p', 'GCG_G_p', 'GCT_G_p', 'TCA_G_p', 'TCC_G_p', 'TCG_G_p', 'TCT_G_p', 'ACA_G_up', 'ACC_G_up', 'ACG_G_up', 'ACT_G_up', 'CCA_G_up', 'CCC_G_up', 'CCG_G_up', 'CCT_G_up', 'GCA_G_up', 'GCC_G_up', 'GCG_G_up', 'GCT_G_up', 'TCA_G_up', 'TCC_G_up', 'TCG_G_up', 'TCT_G_up', 'ACA_T_p', 'ACC_T_p', 'ACG_T_p', 'ACT_T_p', 'CCA_T_p', 'CCC_T_p', 'CCG_T_p', 'CCT_T_p', 'GCA_T_p', 'GCC_T_p', 'GCG_T_p', 'GCT_T_p', 'TCA_T_p', 'TCC_T_p', 'TCG_T_p', 'TCT_T_p', 'ACA_T_up', 'ACC_T_up', 'ACG_T_up', 'ACT_T_up', 'CCA_T_up', 'CCC_T_up', 'CCG_T_up', 'CCT_T_up', 'GCA_T_up', 'GCC_T_up', 'GCG_T_up', 'GCT_T_up', 'TCA_T_up', 'TCC_T_up', 'TCG_T_up', 'TCT_T_up', 'AGA_A_p', 'AGC_A_p', 'AGG_A_p', 'AGT_A_p', 'CGA_A_p', 'CGC_A_p', 'CGG_A_p', 'CGT_A_p', 'GGA_A_p', 'GGC_A_p', 'GGG_A_p', 'GGT_A_p', 'TGA_A_p', 'TGC_A_p', 'TGG_A_p', 'TGT_A_p', 'AGA_A_up', 'AGC_A_up', 'AGG_A_up', 'AGT_A_up', 'CGA_A_up', 'CGC_A_up', 'CGG_A_up', 'CGT_A_up', 'GGA_A_up', 'GGC_A_up', 'GGG_A_up', 'GGT_A_up', 'TGA_A_up', 'TGC_A_up', 'TGG_A_up', 'TGT_A_up', 'AGA_C_p', 'AGC_C_p', 'AGG_C_p', 'AGT_C_p', 'CGA_C_p', 'CGC_C_p', 'CGG_C_p', 'CGT_C_p', 'GGA_C_p', 'GGC_C_p', 'GGG_C_p', 'GGT_C_p', 'TGA_C_p', 'TGC_C_p', 'TGG_C_p', 'TGT_C_p', 'AGA_C_up', 'AGC_C_up', 'AGG_C_up', 'AGT_C_up', 'CGA_C_up', 'CGC_C_up', 'CGG_C_up', 'CGT_C_up', 'GGA_C_up', 'GGC_C_up', 'GGG_C_up', 'GGT_C_up', 'TGA_C_up', 'TGC_C_up', 'TGG_C_up', 'TGT_C_up', 'AGA_T_p', 'AGC_T_p', 'AGG_T_p', 'AGT_T_p', 'CGA_T_p', 'CGC_T_p', 'CGG_T_p', 'CGT_T_p', 'GGA_T_p', 'GGC_T_p', 'GGG_T_p', 'GGT_T_p', 'TGA_T_p', 'TGC_T_p', 'TGG_T_p', 'TGT_T_p', 'AGA_T_up', 'AGC_T_up', 'AGG_T_up', 'AGT_T_up', 'CGA_T_up', 'CGC_T_up', 'CGG_T_up', 'CGT_T_up', 'GGA_T_up', 'GGC_T_up', 'GGG_T_up', 'GGT_T_up', 'TGA_T_up', 'TGC_T_up', 'TGG_T_up', 'TGT_T_up', 'ATA_A_p', 'ATC_A_p', 'ATG_A_p', 'ATT_A_p', 'CTA_A_p', 'CTC_A_p', 'CTG_A_p', 'CTT_A_p', 'GTA_A_p', 'GTC_A_p', 'GTG_A_p', 'GTT_A_p', 'TTA_A_p', 'TTC_A_p', 'TTG_A_p', 'TTT_A_p', 'ATA_A_up', 'ATC_A_up', 'ATG_A_up', 'ATT_A_up', 'CTA_A_up', 'CTC_A_up', 'CTG_A_up', 'CTT_A_up', 'GTA_A_up', 'GTC_A_up', 'GTG_A_up', 'GTT_A_up', 'TTA_A_up', 'TTC_A_up', 'TTG_A_up', 'TTT_A_up', 'ATA_C_p', 'ATC_C_p', 'ATG_C_p', 'ATT_C_p', 'CTA_C_p', 'CTC_C_p', 'CTG_C_p', 'CTT_C_p', 'GTA_C_p', 'GTC_C_p', 'GTG_C_p', 'GTT_C_p', 'TTA_C_p', 'TTC_C_p', 'TTG_C_p', 'TTT_C_p', 'ATA_C_up', 'ATC_C_up', 'ATG_C_up', 'ATT_C_up', 'CTA_C_up', 'CTC_C_up', 'CTG_C_up', 'CTT_C_up', 'GTA_C_up', 'GTC_C_up', 'GTG_C_up', 'GTT_C_up', 'TTA_C_up', 'TTC_C_up', 'TTG_C_up', 'TTT_C_up', 'ATA_G_p', 'ATC_G_p', 'ATG_G_p', 'ATT_G_p', 'CTA_G_p', 'CTC_G_p', 'CTG_G_p', 'CTT_G_p', 'GTA_G_p', 'GTC_G_p', 'GTG_G_p', 'GTT_G_p', 'TTA_G_p', 'TTC_G_p', 'TTG_G_p', 'TTT_G_p', 'ATA_G_up', 'ATC_G_up', 'ATG_G_up', 'ATT_G_up', 'CTA_G_up', 'CTC_G_up', 'CTG_G_up', 'CTT_G_up', 'GTA_G_up', 'GTC_G_up', 'GTG_G_up', 'GTT_G_up', 'TTA_G_up', 'TTC_G_up', 'TTG_G_up', 'TTT_G_up']


def cut_context(context, start):
    return context[2 + start:2 + start + 3]


if __name__ == '__main__':

    '''
        This script visualises the mutation counts for different nucleotide contexts and stores the mutation rates in
        'matrix_factorization/rates_{clade}.csv'.
    '''

    show_plot = False

    # Define clade to be analysed
    clade = '21J'

    # Determine whether mean or median should be used and whether #counts or log(#counts + pseudo_count)
    estimator = ['mean', 'median'][0]
    log = False
    pseudo_count = 0.5

    # Load data
    df = pd.read_csv('/Users/georgangehrn/Desktop/SARS2-mut-fitness-corr/human_data/counts/counts_all_' + clade + '_nonex_syn.csv')
    df['mutated_nt'] = df['nt_mutation'].apply(lambda x: x[-1])

    # Define start of context relative to center nucleotide (-2, -1, or 0)
    start = -1
    df['context'] = df['context'].apply(cut_context, start=start)

    # Convert to log(counts + pseudo_count) if desired
    if log:
        df['actual_count'] = np.log(df['actual_count'].values + pseudo_count)

    # Separate data into paired (p) and unpaired (up)
    df_p = df[df['unpaired'] == 0]
    df_up = df[df['unpaired'] == 1]

    # Load expected counts
    suffix = 'log_' if log else ''
    df_exp = pd.read_csv('/Users/georgangehrn/Desktop/SARS2-mut-fitness-corr/human_data/exp_counts/exp_' + suffix + 'counts_' + clade + '.csv')
    df_exp.set_index('type', inplace=True)

    # Prepare figure
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), dpi=100)

    # Collect mutation counts for all 2*12*4*4 combinations
    mean_p, mean_up = {}, {}
    median_p, median_up = {}, {}
    stdev_p, stdev_up = {}, {}
    med_dev_p, med_dev_up = {}, {}
    N_p, N_up = {}, {}

    for nt2 in letters:  # ancestral nt

        for nt4 in letters:  # mutated nt

            if nt4 != nt2:

                for nt1 in letters:  # left context

                    for nt3 in letters:  # right context

                        # Extract correct context and mutation type
                        df_p_help = df_p[(df_p['context'] == nt1 + nt2 + nt3) * (df_p['mutated_nt'] == nt4)]
                        df_up_help = df_up[(df_up['context'] == nt1 + nt2 + nt3) * (df_up['mutated_nt'] == nt4)]

                        # Get number of datapoints
                        N_p[nt1 + nt2 + nt3 + '_' + nt4] = len(df_p_help)
                        N_up[nt1 + nt2 + nt3 + '_' + nt4] = len(df_up_help)

                        # Get estimate and deviation counts
                        if len(df_p_help) > 0:
                            mean_p[nt1 + nt2 + nt3 + '_' + nt4 + '_p'] = np.mean(df_p_help['actual_count'].values)
                            stdev_p[nt1 + nt2 + nt3 + '_' + nt4 + '_p'] = np.std(df_p_help['actual_count'].values)
                            median_p[nt1 + nt2 + nt3 + '_' + nt4 + '_p'] = np.median(df_p_help['actual_count'].values)
                            med_dev_p[nt1 + nt2 + nt3 + '_' + nt4 + '_p'] = stats.median_abs_deviation(
                                df_p_help['actual_count'].values)
                        else:
                            mean_p[nt1 + nt2 + nt3 + '_' + nt4 + '_p'] = 0
                            stdev_p[nt1 + nt2 + nt3 + '_' + nt4 + '_p'] = 0
                            median_p[nt1 + nt2 + nt3 + '_' + nt4 + '_p'] = 0
                            med_dev_p[nt1 + nt2 + nt3 + '_' + nt4 + '_p'] = 0

                        if len(df_up_help) > 0:
                            mean_up[nt1 + nt2 + nt3 + '_' + nt4 + '_up'] = np.mean(df_up_help['actual_count'].values)
                            stdev_up[nt1 + nt2 + nt3 + '_' + nt4 + '_up'] = np.std(df_up_help['actual_count'].values)
                            median_up[nt1 + nt2 + nt3 + '_' + nt4 + '_up'] = np.median(
                                df_up_help['actual_count'].values)
                            med_dev_up[nt1 + nt2 + nt3 + '_' + nt4 + '_up'] = stats.median_abs_deviation(
                                df_up_help['actual_count'].values)
                        else:
                            mean_up[nt1 + nt2 + nt3 + '_' + nt4 + '_up'] = 0
                            stdev_up[nt1 + nt2 + nt3 + '_' + nt4 + '_up'] = 0
                            median_up[nt1 + nt2 + nt3 + '_' + nt4 + '_up'] = 0
                            med_dev_up[nt1 + nt2 + nt3 + '_' + nt4 + '_up'] = 0

                if estimator == 'median':
                    mean_p, mean_up = median_p, median_up
                    stdev_p, stdev_up = med_dev_p, med_dev_up

                # Filter mutation type (not solved in a very elegant way)
                m_p = {key: value for key, value in mean_p.items() if key[1] == nt2 and key[4] == nt4}
                m_up = {key: value for key, value in mean_up.items() if
                        key[1] == nt2 and key[4] == nt4}
                sd_p = {key: value for key, value in stdev_p.items() if
                        key[1] == nt2 and key[4] == nt4}
                sd_up = {key: value for key, value in stdev_up.items() if
                         key[1] == nt2 and key[4] == nt4}
                n_p = {key: value for key, value in N_p.items() if key[1] == nt2 and key[4] == nt4}
                n_up = {key: value for key, value in N_up.items() if key[1] == nt2 and key[4] == nt4}

                # Extract keys and values
                keys = [s[:3] for s in m_p.keys()]
                values_m_p, values_m_up = list(m_p.values()), list(m_up.values())
                values_sd_p = list(sd_p.values())
                values_sd_up = list(sd_up.values())
                values_n_p = list(n_p.values())
                values_n_up = list(n_up.values())

                # Select correct subplot and plot histograms
                ax = axes[plot_map[nt2 + nt4][0], plot_map[nt2 + nt4][1]]

                # Create bar plot with error bars
                ax.bar(np.arange(16) - 0.2, values_m_p, tick_label=keys, width=0.4, color='blue',
                       alpha=0.7, label='paired')
                ax.bar(np.arange(16) + 0.2, values_m_up, tick_label=keys, width=0.4, color='red',
                       alpha=0.7, label='unpaired')
                ax.errorbar(np.arange(16) - 0.2, values_m_p, yerr=values_sd_p, fmt='none', elinewidth=0.8,
                            capsize=3, color='black')
                ax.errorbar(np.arange(16) + 0.2, values_m_up, yerr=values_sd_up, fmt='none',
                            elinewidth=0.8, capsize=3, color='black')
                ax.set_xticks(np.arange(16), keys, rotation='vertical')

                # Add legend
                if [plot_map[nt2 + nt4][0], plot_map[nt2 + nt4][1]] == [0, 0]:
                    ax.legend()

                # Add label for x-axis
                if plot_map[nt2 + nt4][0] == 2:
                    ax.set_xlabel('context')

                # Add expected counts as horizontal lines
                if estimator == 'mean':
                    ax.axhline(y=df_exp.loc['m_syn_p', nt2 + nt4], color='blue', linestyle='-')
                    ax.axhline(y=df_exp.loc['m_syn', nt2 + nt4], color='black', linestyle='-')
                    ax.axhline(y=df_exp.loc['m_syn_up', nt2 + nt4], color='red', linestyle='-')
                else:
                    ax.axhline(y=df_exp.loc['md_syn_p', nt2 + nt4], color='blue', linestyle='-')
                    ax.axhline(y=df_exp.loc['md_syn', nt2 + nt4], color='black', linestyle='-')
                    ax.axhline(y=df_exp.loc['md_syn_up', nt2 + nt4], color='red', linestyle='-')

                # Add label of y-axis
                if plot_map[nt2 + nt4][1] == 0:
                    if log:
                        ax.set_ylabel(estimator + ' log count')
                    else:
                        ax.set_ylabel(estimator + ' count')

                # Set limits on y-axis
                maximum = np.max((values_m_p, values_m_up))
                minimum = np.min((values_m_p, values_m_up))
                ax.set_ylim(min(1.1 * minimum, 0), maximum * 1.1)

                # Add number of datapoints on top of histogram bars
                for i, (key, value_n_p, value_n_up) in enumerate(zip(keys, values_n_p, values_n_up)):
                    ax.text(i - 0.2, values_m_p[i] + 0.01 * maximum if values_m_p[i] > 0 else 0.01,
                            str(value_n_p), ha='center', va='bottom', color='blue', rotation='vertical', fontsize=8)
                    ax.text(i + 0.2, values_m_up[i] + 0.01 * maximum if values_m_up[i] > 0 else 0.01,
                            str(value_n_up), ha='center', va='bottom', color='red', rotation='vertical', fontsize=8)

                # Set mutation type as title
                ax.set_title(nt2 + ' --> ' + nt4)

    plt.tight_layout()
    plt.show() if show_plot else plt.close()

    # Store mutation rates
    result = []
    for s in row_keys:
        if s in mean_p:
            result.append(mean_p[s])
        else:
            result.append(mean_up[s])
    result = np.array(result) / np.sum(result)
    df = pd.DataFrame(result, index=row_keys)
    df.to_csv('/Users/georgangehrn/Desktop/SARS2-mut-fitness-corr/matrix_factorization/rates_' + clade + '.csv', header=False)
