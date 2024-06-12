import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

plt.rcParams.update({'font.size': 14})

letters = ['A', 'C', 'G', 'T']

mt_type_col = ['blue', 'orange', 'green', 'red', 'purple', 'brown',
               'pink', 'gray', 'cyan', 'magenta', 'lime', 'teal']

plot_map = {'AC': [0, 0], 'CA': [0, 1], 'GA': [0, 2], 'TA': [0, 3],
            'AG': [1, 0], 'CG': [1, 1], 'GC': [1, 2], 'TC': [1, 3],
            'AT': [2, 0], 'CT': [2, 1], 'GT': [2, 2], 'TG': [2, 3]
            }

# TODO: Check if this file is still necessary

if __name__ == '__main__':

    glm_version = ['base', 'p_up', 'l_r', 'l_r_st', 'lr', 'lr_pup'][1]  # type of generalised linear model
    norm = 'l2'  # type of regularization
    alpha_lx = {'l2': [0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 100][2],
                'l1': [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1][2]}['l2']  # regularization strength
    pseudo_count = 0.5

    # Read file with mutation data
    df_mut = pd.read_csv('deer/mut_data_deer.csv')

    # Store learned parameters and log counts for every mutation type
    beta_dic = {}
    mean_log_counts = {}
    pred_log_counts = {}

    # Loop through all mutation types
    for nt_1 in letters:
        for nt_2 in letters:
            if nt_1 != nt_2:

                # Only keep specified mutation type
                df = df_mut[df_mut['nt_mutation'].str.match('^' + nt_1 + '.*' + nt_2 + '$')]

                # Extract actual counts and pairing state
                log_y = np.log(df['actual_count'].values + pseudo_count).reshape(-1, 1)
                unpaired = df['unpaired'].values

                # Prepare data matrix X
                base = np.full(len(log_y), 1)
                columns = [base]
                columns += [unpaired]
                X = np.column_stack(columns)

                # Perform least squares regression
                if norm == 'l1':
                    lasso = Lasso(alpha=alpha_lx, fit_intercept=False)
                    lasso.fit(X, log_y)
                    beta_dic[nt_1 + nt_2] = lasso.coef_.reshape(-1, 1)
                else:
                    beta = np.linalg.inv(X.T @ X + alpha_lx * np.identity(X.shape[1])) @ X.T @ log_y
                    beta_dic[nt_1 + nt_2] = beta

                # Collect mutation counts for all contexts
                df_p_help = df[df['unpaired'] == 0]
                df_up_help = df[df['unpaired'] == 1]
                mean_p = np.mean(np.log(df_p_help['actual_count'].values + pseudo_count)) if len(df_p_help) > 0 else 0
                mean_up = np.mean(np.log(df_up_help['actual_count'].values + pseudo_count)) if len(df_up_help) > 0 else 0
                mean_log_counts[nt_1+nt_2] = [mean_p, mean_up]

                # Predict average log_counts with fitted model
                pred_p = np.zeros(1)
                pred_up = np.zeros(1)

                # Prepare input to model depending on glm type
                x_p = np.array([1, 0]).astype(int)
                x_up = np.array([1, 1]).astype(int)

                # Predict average log_count
                if norm == 'l1':
                    pred_p = lasso.predict(x_p.reshape(-1, 1).T)[0]
                    pred_up = lasso.predict(x_up.reshape(-1, 1).T)[0]
                else:
                    pred_p = (beta.T @ x_p)[0]
                    pred_up = (beta.T @ x_up)[0]
                    pred_log_counts[nt_1+nt_2] = [pred_p, pred_up]

    # Plot mean absolute/log counts for all mutation types in one plot
    keys = list(mean_log_counts.keys())
    values = list(mean_log_counts.values())
    paired_values = [val[0] for val in values]
    unpaired_values = [val[1] for val in values]
    pred_values = list(pred_log_counts.values())
    pred_paired_values = [val[0] for val in pred_values]
    pred_unpaired_values = [val[1] for val in pred_values]
    bar_width = 0.4

    r1 = range(len(keys))
    r2 = [x + bar_width for x in r1]

    plt.figure(dpi=200)

    plt.bar(r1, paired_values, color='b', width=bar_width, label='paired', alpha=0.6)
    plt.bar(r2, unpaired_values, color='r', width=bar_width, label='unpaired', alpha=0.6)
    plt.bar(r1, pred_paired_values, width=bar_width, color='none',
            edgecolor='black', hatch="//")
    plt.bar(r2, pred_unpaired_values, width=bar_width, color='none',
            edgecolor='black', label='predicted', hatch="//")

    plt.xlabel('mutation type', fontsize=12)
    plt.ylabel('mean log mutation count', fontsize=12)
    plt.axhline(y=np.log(pseudo_count), linestyle='--', color='black', label='ln(' + str(pseudo_count) + ')')
    plt.axhline(y=0, linestyle='-', color='black')
    plt.xticks([r + bar_width / 2 for r in range(len(keys))], keys, rotation='vertical')

    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)

    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.show()

    # Prepare beta as numpy array with normalised parameters
    beta_arr = np.stack(list(beta_dic.values()), axis=0).reshape(12, 2)
    beta_arr = np.insert(beta_arr, 1, 0, axis=1)
    beta_arr[:, 0] += np.mean(beta_arr[:, 1:3], axis=1)
    beta_arr[:, 1:3] -= np.mean(beta_arr[:, 1:3], axis=1).reshape(-1, 1)
    beta_min, beta_max = np.min(beta_arr[:, 1:]), np.max(beta_arr[:, 1:])
    cutoff = 0.01
    sparse_count = np.count_nonzero(np.abs(beta_arr[:, 1:]) < cutoff)


    # Make a plot for every element in beta (#ofplots = #ofparameters)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), dpi=200)
    axes = axes.flatten()
    titles = [r'$\tilde{\alpha}_{base}$', r'$\tilde{\alpha}_{p}$', r'$\tilde{\alpha}_{up}$']
    for i in range(0, 3):
        ax = axes[i]
        ax.axhline(0, color='black')
        ax.set_title(titles[i], fontsize=10)
        values = beta_arr[:, i]
        keys = beta_dic.keys()
        positions = np.arange(len(keys))
        ax.bar(positions, np.array(values).flatten(), align='center', width=0.8,
               color=[mt_type_col[i] for i in range(12)])
        ax.set_xticks(positions)
        ax.set_xticklabels(keys, rotation='vertical')
        ax.set_xlabel('mutation type')
        if i > 0:
            ax.set_ylim(1.05 * beta_min, 1.05 * beta_max)
    plt.tight_layout()
    plt.show()
