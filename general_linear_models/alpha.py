import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from helper import load_data

letters = ['A', 'C', 'G', 'T']

mt_type_col = ['blue', 'orange', 'green', 'red', 'purple', 'brown',
               'pink', 'gray', 'cyan', 'magenta', 'lime', 'teal']

plot_map = {'AC': [0, 0], 'CA': [0, 1], 'GA': [0, 2], 'TA': [0, 3],
            'AG': [1, 0], 'CG': [1, 1], 'GC': [1, 2], 'TC': [1, 3],
            'AT': [2, 0], 'CT': [2, 1], 'GT': [2, 2], 'TG': [2, 3]
            }

# TODO: Integrate this functionality into fit_general_linear_model.py and delete this file

def cut_context(context, start):
    return context[2 + start:2 + start + 3]


def get_left_context(context, start):
    index = 1 if start == 0 else 0
    return context[index]


def get_right_context(context, start):
    index = -2 if start == -2 else -1
    return context[index]


def get_context(nt_l, nt_r, nt_1, start):
    context = nt_l + nt_1 + nt_r
    if start == -2:
        context = nt_l + nt_r + nt_1
    if start == 0:
        context = nt_1 + nt_l + nt_r

    return context


def one_hot_l_r(context_l, context_r):
    sigma = {}
    for nt in ['C', 'G', 'T']:
        sigma[nt + '_l'] = (context_l == nt).astype(int)
        sigma[nt + '_r'] = (context_r == nt).astype(int)

    return [sigma['C_l'], sigma['G_l'], sigma['T_l'],
            sigma['C_r'], sigma['G_r'], sigma['T_r']]


def one_hot_l_r_st(context_l, context_r, unpaired):
    sigma = {}
    for nt in letters:
        for state in [0, 1]:
            st = 'p' if state == 0 else 'up'
            sigma[nt + '_l_' + st] = ((context_l == nt) * (unpaired == state)).astype(int)
            sigma[nt + '_r_' + st] = ((context_r == nt) * (unpaired == state)).astype(int)

    return [sigma['A_l_up'], sigma['C_l_p'], sigma['C_l_up'], sigma['G_l_p'], sigma['G_l_up'],
            sigma['T_l_p'], sigma['T_l_up'], sigma['A_r_up'], sigma['C_r_p'], sigma['C_r_up'],
            sigma['G_r_p'], sigma['G_r_up'], sigma['T_r_p'], sigma['T_r_up']]


def one_hot_lr(context_l, context_r):
    sigma = {}
    for nt1 in letters:
        for nt2 in letters:
            sigma[nt1 + nt2] = (context_l + context_r == nt1 + nt2).astype(int)

    return [sigma['AC'], sigma['AG'], sigma['AT'], sigma['CA'], sigma['CC'], sigma['CG'],
            sigma['CT'], sigma['GA'], sigma['GC'], sigma['GG'], sigma['GT'], sigma['TA'],
            sigma['TC'], sigma['TG'], sigma['TT']]


def one_hot_lr_pup(context_l, context_r, unpaired):
    sigma = {}
    for nt1 in letters:
        for nt2 in letters:
            for state in [0, 1]:
                st = 'p' if state == 0 else 'up'
                sigma[nt1 + nt2 + '_' + st] = ((context_l + context_r == nt1 + nt2) * (unpaired == state)).astype(int)

    return [sigma['AA_up'], sigma['AC_p'], sigma['AC_up'], sigma['AG_p'], sigma['AG_up'],
            sigma['AT_p'], sigma['AT_up'], sigma['CA_p'], sigma['CA_up'], sigma['CC_p'], sigma['CC_up'],
            sigma['CG_p'], sigma['CG_up'], sigma['CT_p'], sigma['CT_up'], sigma['GA_p'], sigma['GA_up'],
            sigma['GC_p'], sigma['GC_up'], sigma['GG_p'], sigma['GG_up'], sigma['GT_p'], sigma['GT_up'],
            sigma['TA_p'], sigma['TA_up'], sigma['TC_p'], sigma['TC_up'], sigma['TG_p'], sigma['TG_up'],
            sigma['TT_p'], sigma['TT_up']]


if __name__ == '__main__':

    # Define parameters
    start = -1  # start of tri-nt context (-2, -1, or 0) relative to mutating nt x in abxcd
    glm_versions = ['base', 'p_up', 'l_r', 'l_r_st', 'lr', 'lr_pup']  # type of generalised linear model
    norm = 'l2'  # type of regularization
    alphas_l2 = [0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 100]  # l2 regularization strength
    alphas_l1 = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1]  # l1 regularization strength
    plot_counts = True  # plot observed and predicted count for every mutation type and context
    plot_beta = True  # plot parameters learned during least squares regression
    clade = '21J'

    # Prepare container to store mean squared errors of different general_linear_models versions / different
    # regularization strengths
    mse_dic = {'AC': np.array([]), 'AG': np.array([]), 'AT': np.array([]),
               'CA': np.array([]), 'CG': np.array([]), 'CT': np.array([]),
               'GA': np.array([]), 'GC': np.array([]), 'GT': np.array([]),
               'TA': np.array([]), 'TC': np.array([]), 'TG': np.array([])}

    # Loop over all selected regularization strengths
    for alpha_lx in (alphas_l1[2:3] if norm == 'l1' else alphas_l2[2:3]):

        # Loop over all selected general_linear_models versions
        for glm_version in glm_versions[5:6]:

            # Store learned parameters for every mutation type
            beta_dic = {}

            fig, axes = plt.subplots(3, 4, figsize=(16, 10), dpi=200)

            # Loop through all mutation types
            for nt_1 in letters:
                for nt_2 in letters:
                    if nt_1 != nt_2:

                        # Load data for mutation nt_1 --> nt_2
                        df = load_data(nt_x=nt_1, nt_y=nt_2, pseudo_count=0.5, clade=clade,
                                       cell_type='Huh7', verbose=False)

                        # Cut out chosen context
                        df['context'] = df['context'].apply(cut_context, start=start)

                        # Extract actual counts, pairing state, and context
                        log_y = np.log(df['actual_count'].values).reshape(-1, 1)
                        unpaired = df['unpaired'].values
                        context_l = df['context'].apply(get_left_context, start=start).values
                        context_r = df['context'].apply(get_right_context, start=start).values

                        # Prepare data matrix X
                        base = np.full(len(log_y), 1)
                        columns = [base]
                        if glm_version == 'p_up':
                            columns += [unpaired]
                        if glm_version == 'l_r':
                            columns += [unpaired] + one_hot_l_r(context_l, context_r)
                        if glm_version == 'l_r_st':
                            columns += [unpaired] + one_hot_l_r_st(context_l, context_r, unpaired)
                        if glm_version == 'lr':
                            columns += [unpaired] + one_hot_lr(context_l, context_r)
                        if glm_version == 'lr_pup':
                            columns += one_hot_lr_pup(context_l, context_r, unpaired)
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
                        mean_p, mean_up = {}, {}
                        df_p = df[df['unpaired'] == 0]
                        df_up = df[df['unpaired'] == 1]
                        for nt_l in letters:
                            for nt_r in letters:
                                # Get context
                                context = get_context(nt_l, nt_r, nt_1, start)
                                # Select data for this context
                                df_p_help = df_p[(df_p['context'] == context)]
                                df_up_help = df_up[(df_up['context'] == context)]
                                # Calculate mean log_count for this context
                                mean_p[context] = np.mean(np.log(df_p_help['actual_count'].values)) if len(
                                    df_p_help) > 0 else 0
                                mean_up[context] = np.mean(np.log(df_up_help['actual_count'].values)) if len(
                                    df_up_help) > 0 else 0

                        # Extract keys (16 different contexts) and values (corresponding mean log_counts)
                        keys = [s for s in mean_p.keys()]
                        m_p, m_up = list(mean_p.values()), list(mean_up.values())

                        # Predict average log_counts with fitted model
                        pred_p = np.zeros(16)
                        pred_up = np.zeros(16)
                        for i, key in enumerate(keys):

                            # Get context
                            nt_l, nt_r = get_left_context(key, start), get_right_context(key, start)

                            # Prepare input to model depending on glm type
                            x_p = np.array([1])
                            x_up = np.array([1])
                            if glm_version == 'p_up':
                                x_p = np.array([1, 0]).astype(int)
                                x_up = np.array([1, 1]).astype(int)
                            if glm_version == 'l_r':
                                x_p = np.array([1, 0, nt_l == 'C', nt_l == 'G', nt_l == 'T',
                                                nt_r == 'C', nt_r == 'G', nt_r == 'T']).astype(int)
                                x_up = np.array([1, 1, nt_l == 'C', nt_l == 'G', nt_l == 'T',
                                                 nt_r == 'C', nt_r == 'G', nt_r == 'T']).astype(int)
                            if glm_version == 'l_r_st':
                                x_p = np.array([1, 0, 0, nt_l == 'C', 0, nt_l == 'G', 0, nt_l == 'T', 0,
                                                0, nt_r == 'C', 0, nt_r == 'G', 0, nt_r == 'T', 0]).astype(int)
                                x_up = np.array([1, 1, nt_l == 'A', 0, nt_l == 'C', 0, nt_l == 'G', 0, nt_l == 'T',
                                                 nt_r == 'A', 0, nt_r == 'C', 0, nt_r == 'G', 0, nt_r == 'T']).astype(
                                    int)
                            if glm_version == 'lr':
                                k = nt_l + nt_r
                                x_p = np.array([1, 0, k == 'AC', k == 'AG', k == 'AT', k == 'CA', k == 'CC',
                                                k == 'CG', k == 'CT', k == 'GA', k == 'GC', k == 'GG', k == 'GT',
                                                k == 'TA', k == 'TC', k == 'TG', k == 'TT']).astype(int)
                                x_up = np.array([1, 1, k == 'AC', k == 'AG', k == 'AT', k == 'CA', k == 'CC',
                                                 k == 'CG', k == 'CT', k == 'GA', k == 'GC', k == 'GG', k == 'GT',
                                                 k == 'TA', k == 'TC', k == 'TG', k == 'TT']).astype(int)
                            if glm_version == 'lr_pup':
                                k = nt_l + nt_r
                                x_p = np.array(
                                    [1, 0, k == 'AC', 0, k == 'AG', 0, k == 'AT', 0, k == 'CA', 0, k == 'CC', 0,
                                     k == 'CG', 0, k == 'CT', 0, k == 'GA', 0, k == 'GC', 0, k == 'GG', 0, k == 'GT',
                                     0, k == 'TA', 0, k == 'TC', 0, k == 'TG', 0, k == 'TT', 0]).astype(int)
                                x_up = np.array([1, k == 'AA', 0, k == 'AC', 0, k == 'AG', 0, k == 'AT', 0,
                                                 k == 'CA', 0, k == 'CC', 0, k == 'CG', 0, k == 'CT', 0, k == 'GA',
                                                 0, k == 'GC', 0, k == 'GG', 0, k == 'GT', 0, k == 'TA', 0, k == 'TC',
                                                 0, k == 'TG', 0, k == 'TT']).astype(int)

                            # Predict average log_count
                            if norm == 'l1':
                                pred_p[i] = lasso.predict(x_p.reshape(-1, 1).T)[0]
                                pred_up[i] = lasso.predict(x_up.reshape(-1, 1).T)[0]
                            else:
                                pred_p[i] = (beta.T @ x_p)[0]
                                pred_up[i] = (beta.T @ x_up)[0]

                        # Add data to plot
                        if plot_counts:
                            # Choose subplot according to mutation type
                            ax = axes[plot_map[nt_1 + nt_2][0], plot_map[nt_1 + nt_2][1]]

                            # Set x and y labels
                            ax.set_title(nt_1 + ' --> ' + nt_2, fontsize=10)
                            if plot_map[nt_1 + nt_2][0] == 2:
                                ax.set_xlabel('context', fontsize=9)
                            if plot_map[nt_1 + nt_2][1] == 0:
                                ax.set_ylabel('mean log count', fontsize=9)
                            ax.tick_params(axis='y', labelsize=7)

                            # Add data and predictions of model
                            ax.bar(np.arange(16) - 0.2, m_p, tick_label=keys, width=0.4, color='blue', alpha=0.7,
                                   label='paired')
                            ax.bar(np.arange(16) + 0.2, m_up, tick_label=keys, width=0.4, color='red', alpha=0.7,
                                   label='unpaired')
                            ax.bar(np.arange(16) - 0.2, pred_p, tick_label=keys, width=0.4, color='none',
                                   edgecolor='black', alpha=0.7, label='paired', hatch="//")
                            ax.bar(np.arange(16) + 0.2, pred_up, tick_label=keys, width=0.4, color='none',
                                   edgecolor='black', alpha=0.7, label='unpaired', hatch="//")
                            ax.set_xticks(np.arange(16), keys, rotation='vertical', fontsize=7)

                            # Set limits for y-axis
                            maximum = np.max((m_p, m_up, pred_p, pred_up))
                            minimum = np.min((m_p, m_up, pred_p, pred_up))
                            ax.set_ylim(min(1.1 * minimum, 0), maximum * 1.1)

                            # Add learned parameters and mean squared error
                            if norm == 'l1':
                                mse = np.linalg.norm(log_y.flatten() - lasso.predict(X)) ** 2 / len(log_y)
                                ax.text(-1, 1 * maximum, 'mse = ' + str(np.round(mse, 2)) +
                                        ', l1 reg. = ' + str(alpha_lx) +
                                        ', general_linear_models type: ' + glm_version, fontsize=6)
                                ax.text(-1, 1.05 * maximum, 'alpha = ' + str(np.round(lasso.coef_.flatten(), 2)),
                                        fontsize=6)
                            else:
                                mse = np.linalg.norm(log_y - X @ beta) ** 2 / len(log_y)
                                ax.text(-1, 1.0 * maximum, 'mse = ' + str(round(mse, 2)) +
                                        ', l2 reg. = ' + str(alpha_lx) +
                                        ', general_linear_models type: ' + glm_version, fontsize=6)
                                ax.text(-1, 1.05 * maximum, 'alpha = ' + str(np.round(beta.flatten(), 2)),
                                        fontsize=6)

                        # Add mse to dictionary
                        if norm == 'l1':
                            mse = np.linalg.norm(log_y.flatten() - lasso.predict(X)) ** 2 / len(log_y)
                        else:
                            mse = np.linalg.norm(log_y - X @ beta) ** 2 / len(log_y)
                        mse_dic[nt_1 + nt_2] = np.append(mse_dic[nt_1 + nt_2], mse)

            # Plot data versus predictions
            if plot_counts:
                plt.tight_layout()
                plt.savefig('model-vs-pred_all-nt.png')
                plt.show()
            else:
                plt.close()

            # Prepare beta as numpy array with normalised parameters
            beta_arr = np.stack(list(beta_dic.values()), axis=0).reshape(12, 8)
            beta_arr = np.insert(beta_arr, [1, 2, 5], 0, axis=1)
            beta_arr[:, 0] += np.mean(beta_arr[:, 1:3], axis=1)
            beta_arr[:, 1:3] -= np.mean(beta_arr[:, 1:3], axis=1).reshape(-1, 1)
            beta_arr[:, 0] += np.mean(beta_arr[:, 3:7], axis=1)
            beta_arr[:, 3:7] -= np.mean(beta_arr[:, 3:7], axis=1).reshape(-1, 1)
            beta_arr[:, 0] += np.mean(beta_arr[:, 7:11], axis=1)
            beta_arr[:, 7:11] -= np.mean(beta_arr[:, 7:11], axis=1).reshape(-1, 1)
            beta_min, beta_max = np.min(beta_arr[:, 1:]), np.max(beta_arr[:, 1:])
            cutoff = 0.01
            sparse_count = np.count_nonzero(np.abs(beta_arr[:, 1:]) < cutoff)
            print(sparse_count)

            np.savetxt('results/21J/alpha_21J.csv', beta_arr, delimiter=',')

        if plot_beta:

            # Make a plot for every element in beta (#ofplots = #ofparameters)
            fig, axes = plt.subplots(3, 4, figsize=(16, 10), dpi=200)
            axes = axes.flatten()
            titles = [r'$\tilde{\alpha}_{base}$', r'$\tilde{\alpha}_{p}$', r'$\tilde{\alpha}_{up}$',
                      r'$\tilde{\alpha}_{A,l}$', r'$\tilde{\alpha}_{C,l}$', r'$\tilde{\alpha}_{G,l}$',
                      r'$\tilde{\alpha}_{T,l}$', r'$\tilde{\alpha}_{A,r}$', r'$\tilde{\alpha}_{C,r}$',
                      r'$\tilde{\alpha}_{G,r}$', r'$\tilde{\alpha}_{T,r}$']
            axes[3].axis('off')
            axes[3].text(0.1, 0.5, norm + ' reg. = ' + str(alpha_lx) + '\ngeneral_linear_models type: ' + glm_version +
                         '\n#ofparams <' + str(cutoff) + ': ' + str(sparse_count) + '/' + str(np.size(beta_arr[:, 1:])),
                         fontsize=14)
            for i in range(0, 11):
                if i < 3:
                    ax = axes[i]
                else:
                    ax = axes[i+1]
                ax.axhline(0, color='black')
                ax.set_title(titles[i], fontsize=10)
                values = beta_arr[:, i]
                keys = beta_dic.keys()
                positions = np.arange(len(keys))
                ax.bar(positions, np.array(values).flatten(), align='center', width=0.8,
                       color=[mt_type_col[i] for i in range(12)])
                ax.set_xticks(positions)
                ax.set_xticklabels(keys, rotation='vertical')
                if (i+1) % 4 != 0 and i != 0 and i != 1:
                    ax.set_yticklabels([])
                if i > 6:
                    ax.set_xlabel('mutation type')
                if i > 0:
                    ax.set_ylim(1.05 * beta_min, 1.05 * beta_max)
            plt.tight_layout()
            plt.savefig('alpha_1.png')
            plt.show()

            # Make a plot of beta for every mutation type (12 plots)
            fig, axes = plt.subplots(3, 4, figsize=(16, 10), dpi=200)
            axes = axes.flatten()
            ax_inv = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11,]
            for i, (key, values) in enumerate(beta_dic.items()):
                ax = axes[ax_inv[i]]
                ax.axhline(0, color='black')
                values = beta_arr[i, 1:]
                positions = np.arange(len(values))
                ax.bar(positions, values, align='center', width=0.8, color=mt_type_col[i])
                ax.set_title(key[0] + ' --> ' + key[1], fontsize=10)
                ax.set_xticks(positions)  # Set the positions of x-ticks
                ax.set_xticklabels(titles[1:])
                if ax_inv[i] % 4 != 0:
                    ax.set_yticks([])
                ax.set_ylim(1.05 * beta_min, 1.05 * beta_max)
            plt.tight_layout()
            plt.savefig('alpha_2.png')
            plt.show()

    # Plot mse over different general_linear_models versions / regularization strengths
    if len(mse_dic['AC']) > 1:
        mse_mean = np.zeros(len(mse_dic['AC']))
        x = np.arange(len(mse_dic['AC'])) + 1
        for i, (key, value) in enumerate(mse_dic.items()):
            mse_mean += value
            plt.plot(x, value, 'o--', label=key[0] + '-->' + key[1], color=mt_type_col[i % len(mt_type_col)])
        plt.plot(x, mse_mean / 12, 'o-', linewidth=2, markersize=8, label='mean', color='black')
        plt.xlabel('version (cf. map on the right)')
        plt.ylim(0.2)
        plt.title(['aXY --> bXY', 'XaY --> XbY', 'XYa --> XYb'][-start])
        plt.ylabel('mean squared error')
        plt.legend(loc='upper right')
        plt.show()
