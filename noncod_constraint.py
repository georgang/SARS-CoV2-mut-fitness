from helper import load_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

    df = load_data(nt_x='all', nt_y='all', pseudo_count=0.5, clade='21J', sites='syn',
                   cell_type='Huh7', verbose=False)

    alpha = np.loadtxt("results/21J/alpha_21J.csv", delimiter=",")
    alpha_index = (df['nt_type'] + df['mutated_nt']).map(alpha_map).values

    nt_site = df['nt_site'].values
    act_log_count = np.log(df['actual_count']).values  # pseudo count 0.5 already added in line above
    unpaired = df['unpaired'].values
    context_l = df['context'].apply(get_left_context).map(nt_map)  # assumes that we look at -1/+1 context
    context_r = df['context'].apply(get_right_context).map(nt_map)

    n = len(unpaired)
    x = np.zeros((n, 11))  # There are 11 parameters
    x[:, 0] = 1
    x[np.arange(n), 1 + unpaired] = 1
    x[np.arange(n), 3 + context_l] = 1
    x[np.arange(n), 7 + context_r] = 1

    pred_log_count = np.diagonal(np.matmul(x, alpha[alpha_index].T))  # prediction of log(counts + 0.5)

    exp_log_count = pd.read_csv('human_data/exp_counts/exp_log_counts_21J.csv')
    exp_log_count = exp_log_count[(df['nt_type'] + df['mutated_nt'])].values[0]

    fitness_corr = act_log_count - pred_log_count
    fitness_uncorr = act_log_count - exp_log_count

    df_uncorr = pd.DataFrame({'x': nt_site, 'y': fitness_uncorr})
    df_uncorr = df_uncorr.groupby('x', as_index=False)['y'].mean()
    window_size = 10
    rolling_mean = df_uncorr.rolling(window=window_size).mean()
    df_uncorr = rolling_mean.dropna()

    df_corr = pd.DataFrame({'x': nt_site, 'y': fitness_corr})
    df_corr = df_corr.groupby('x', as_index=False)['y'].mean()
    rolling_mean = df_corr.rolling(window=window_size).mean()
    df_corr = rolling_mean.dropna()

    fig, ax = plt.subplots(figsize=(20, 5))

    x_corr, y_corr = df_corr['x'].values, df_corr['y'].values
    x_uncorr, y_uncorr = df_uncorr['x'].values, df_uncorr['y'].values
    ax.plot(x_corr, y_corr, linestyle='-')
    ax.plot(x_uncorr, y_uncorr, linestyle='-')
    ax.axhline(0, color='k', linestyle='-')
    plt.show()

    fig, ax = plt.subplots(figsize=(20, 5))

    ax.plot(x_corr[x_corr > 25000], y_corr[x_corr > 25000], linestyle='-')
    ax.plot(x_uncorr[x_corr > 25000], y_uncorr[x_uncorr > 25000], linestyle='-')
    ax.axhline(0, color='k', linestyle='-')
    plt.show()