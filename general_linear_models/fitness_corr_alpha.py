import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper import load_data

plt.rcParams.update({'font.size': 14})

letters = ['A', 'C', 'G', 'T']

alpha_map = {'AC': 0, 'AG': 1, 'AT': 2, 'CA': 3, 'CG': 4, 'CT': 5,
             'GA': 6, 'GC': 7, 'GT': 8, 'TA': 9, 'TC': 10, 'TG': 11}

nt_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

plot_map = {'AC': [0, 0], 'CA': [0, 1], 'GA': [0, 2], 'TA': [0, 3],
            'AG': [1, 0], 'CG': [1, 1], 'GC': [1, 2], 'TC': [1, 3],
            'AT': [2, 0], 'CT': [2, 1], 'GT': [2, 2], 'TG': [2, 3]}

# TODO: Improve this

def get_left_context(context):
    return context[1]


def get_right_context(context):
    return context[3]


if __name__ == '__main__':

    fig, axes = plt.subplots(3, 4, figsize=(16, 10), dpi=200)

    alpha_all = np.loadtxt("results/21J/alpha_21J.csv", delimiter=",")

    fitness_uncorr_dic = {}
    fitness_corr_dic = {}
    fitness_corr_2_dic = {}

    for nt_1 in letters:

        for nt_2 in letters:

            if nt_1 != nt_2:

                alpha = alpha_all[alpha_map[nt_1 + nt_2], :]  # TODO: Check that this map is correct
                alpha_2 = alpha_all[alpha_map[nt_1 + nt_2], :3]  # TODO: Check that this map is correct

                df = load_data(nt_x=nt_1, nt_y=nt_2, pseudo_count=0.5, clade='21J', sites='syn',
                               cell_type='Huh7', verbose=False)

                act_log_count = np.log(df['actual_count']).values  # pseudo count 0.5 already added in line above
                unpaired = df['unpaired'].values
                context_l = df['context'].apply(get_left_context).map(nt_map)  # assumes that we look at -1/+1 context
                context_r = df['context'].apply(get_right_context).map(nt_map)  # assumes that we look at -1/+1 context

                n = len(unpaired)
                x = np.zeros((n, 11))  # There are 11 parameters
                x_2 = np.zeros((n, 3))  # There are 11 parameters
                x[:, 0] = 1
                x_2[:, 0] = 1
                x[np.arange(n), 1 + unpaired] = 1
                x_2[np.arange(n), 1 + unpaired] = 1
                x[np.arange(n), 3 + context_l] = 1
                x[np.arange(n), 7 + context_r] = 1

                pred_log_count = np.matmul(x, alpha.reshape(-1, 1)).flatten()  # prediction of log(counts + 0.5)
                pred_log_count_2 = np.matmul(x_2, alpha_2.reshape(-1, 1)).flatten()  # prediction of log(counts + 0.5)

                exp_log_count = pd.read_csv('../human_data/exp_counts/exp_log_counts_21J.csv')
                exp_log_count = np.full(len(unpaired), exp_log_count[nt_1 + nt_2][0])  # mean log(count + 0.5)

                ax = axes[plot_map[nt_1 + nt_2][0], plot_map[nt_1 + nt_2][1]]
                ax.set_title(nt_1 + ' --> ' + nt_2)
                ax.text(0.7, 0.6, 'n = ' + str(n), transform=ax.transAxes, fontsize=12, color='black')

                fitness_uncorr = act_log_count - exp_log_count
                fitness_uncorr_dic[nt_1 + nt_2] = fitness_uncorr
                var_uncorr = np.var(fitness_uncorr)
                ax.hist(fitness_uncorr, bins=30, alpha=0.7, color='black', label='uncorrected')
                ax.text(0.7, 0.5, 'var: ' + str(round(var_uncorr, 2)), transform=ax.transAxes,
                        fontsize=12, color='black')

                fitness_corr = act_log_count - pred_log_count
                fitness_corr_dic[nt_1 + nt_2] = fitness_corr
                var_corr = np.var(fitness_corr)
                ax.hist(fitness_corr, bins=30, alpha=0.7, color='green', label='corrected')
                ax.text(0.7, 0.4, 'var: ' + str(round(var_corr, 2)), transform=ax.transAxes,
                        fontsize=12, color='green')

                fitness_corr_2 = act_log_count - pred_log_count_2
                fitness_corr_2_dic[nt_1 + nt_2] = fitness_corr_2
                var_corr_2 = np.var(fitness_corr_2)
                ax.hist(fitness_corr_2, bins=30, alpha=0.7, color='red', label='corrected')
                ax.text(0.7, 0.3, 'var: ' + str(round(var_corr_2, 2)), transform=ax.transAxes,
                        fontsize=12, color='red')

                if plot_map[nt_1 + nt_2][0] == 2:
                    ax.set_xlabel('fitness effect')

                if [plot_map[nt_1 + nt_2][0], plot_map[nt_1 + nt_2][1]] == [0, 0]:
                    ax.legend()

    plt.tight_layout()
    plt.show()

    fitness_uncorr = np.concatenate(list(fitness_uncorr_dic.values()))
    fitness_corr = np.concatenate(list(fitness_corr_dic.values()))
    fitness_corr_2 = np.concatenate(list(fitness_corr_2_dic.values()))

    fig, ax = plt.subplots()

    ax.hist(fitness_uncorr + 0.47, bins=40, alpha=0.7, color='black', label='uncorrected')
    ax.hist(fitness_corr, bins=40, alpha=0.7, color='green', label='corrected')

    ax.text(0.7, 0.6, 'var: ' + str(round(np.var(fitness_uncorr), 3)), transform=ax.transAxes,
            fontsize=12, color='black')
    ax.text(0.7, 0.5, 'var: ' + str(round(np.var(fitness_corr), 3)), transform=ax.transAxes,
            fontsize=12, color='green')
    ax.text(0.7, 0.4, 'mean: ' + str(round(np.mean(fitness_uncorr), 2)) + ' $\cdot10^{-4}$',
            transform=ax.transAxes, fontsize=12, color='black')
    ax.text(0.7, 0.3, 'mean: ' + str(round(10000 * np.mean(fitness_corr), 2)) + ' $\cdot10^{-4}$',
            transform=ax.transAxes, fontsize=12, color='green')
    # ax.text(0.7, 0.5, 'mean: ' + str(round(np.mean(fitness_uncorr), 2)),
    #         transform=ax.transAxes, fontsize=12, color='black')
    # ax.text(0.7, 0.4, 'mean: ' + str(round(np.mean(fitness_corr), 2)),
    #         transform=ax.transAxes, fontsize=12, color='green')

    ax.axvline(0, color='grey', linestyle='--')
    ax.set_xlabel('fitness effect')
    ax.set_ylabel('# of sites')
    ax.set_title('fitness effect of synonymous mutations in 21J\n(aggregated over all mutation types)')
    ax.set_xlim((-5, 5))
    plt.legend()
    plt.show()
