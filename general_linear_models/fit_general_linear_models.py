import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from helper import load_mut_counts, plot_map
from sklearn.metrics import mean_squared_error

plt.rcParams.update({'font.size': 12})

letters = ['A', 'C', 'G', 'T']

mut_types = ['AC', 'AG', 'AT', 'CA', 'CG', 'CT', 'GA', 'GC', 'GT', 'TA', 'TC', 'TG']

contexts = ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']

mut_type_cols = ['blue', 'orange', 'green', 'red', 'purple', 'brown',
                 'pink', 'gray', 'cyan', 'magenta', 'lime', 'teal']


def add_predictions(dataframe, clade):
    gen_lin_model = GeneralLinearModel(type='l_r', test_data=dataframe)

    gen_lin_model.W = pd.read_csv(
        f"/Users/georgangehrn/Desktop/SARS-CoV2-mut-fitness/general_linear_models/results/{clade}"
        f"/l_r/learned_params.csv").to_dict(orient='list')

    pred_log_counts = gen_lin_model.predict_log_counts()

    dataframe['pred_log_count'] = pred_log_counts

    return dataframe


class GeneralLinearModel:

    def __init__(self, type, training_data=None, regularization=None, W=None, test_data=None):
        self.type = type  # ['base', 'p_up', 'l_r', 'l_r_st', 'lr', 'lr_pup']
        self.df_train = training_data
        self.reg_type = regularization[0] if regularization is not None else None
        self.reg_strength = regularization[1] if regularization is not None else None
        self.W = W
        self.df_test = test_data
        self.mean_sq_errs = {}
        self.W_l_r_normalized = None

    def train(self):

        self.W = {}

        # Check which mutation types are present in the data
        present_mut_types = self.df_train['nt_mutation'].apply(lambda x: x[0] + x[-1])

        for mut_type in present_mut_types.unique():

            nt1, nt2 = mut_type[0], mut_type[1]

            df_local = self.df_train[self.df_train['nt_mutation'].str.match('^' + nt1 + '.*' + nt2 + '$')]

            log_counts = np.log(df_local['actual_count'].values + 0.5).reshape(-1, 1)  # dimensions (# of sites, 1)

            X, _ = self.create_data_matrix(df_local.copy())  # (# of sites, # of parameters in model), _

            if self.reg_type == 'l1':
                lasso = Lasso(alpha=self.reg_strength, fit_intercept=False)
                lasso.fit(X, log_counts)
                self.W[mut_type] = lasso.coef_  # TODO: Make sure that W @ X is the same as prediction with Lasso
            elif self.reg_type == 'l2':
                w = np.linalg.inv(X.T @ X + self.reg_strength * np.identity(X.shape[1])) @ X.T @ log_counts
                self.W[mut_type] = w

            self.mean_sq_errs[mut_type] = mean_squared_error(log_counts, (X @ self.W[mut_type]).flatten())

        # Store learned parameters in directory
        params = {key: value.flatten() for key, value in self.W.items()}
        params_df = pd.DataFrame(params)

        directory = f"results/{CLADE}/{self.type}"

        if not os.path.exists(directory):
            os.makedirs(directory)

        filepath = os.path.join(directory, 'learned_params.csv')
        params_df.to_csv(filepath, index=False, header=True)

        # Store regularization parameters
        with open(os.path.join(directory, 'regularization_params.log'), 'w') as log_file:
            log_file.write(f"regularization type: {self.reg_type}\n")
            log_file.write(f"regularization strength: {self.reg_strength}\n")

        # Plot observed vs. predicted mean log(counts + 0.5) for every context and pairing state
        self.plot_fit_vs_data()

        # Visualise learned parameters of model of choice
        if self.type == 'l_r':
            self.plot_l_r_params(normalize=False)

    def plot_l_r_params(self, normalize, y_lims=None):

        # Convert dictionary to matrix of dimension (12,8) where rows are ordered AC to TG
        W_matrix = np.vstack([self.W[mut_type].T for mut_type in mut_types])

        # Add 3 additional columns (because of base case)
        W_matrix = np.insert(W_matrix, [1, 2, 5], 0, axis=1)

        if normalize:
            if self.W_l_r_normalized is None:
                # Normalize parameters
                W_matrix[:, 0] += np.mean(W_matrix[:, 1:3], axis=1)
                W_matrix[:, 1:3] -= np.mean(W_matrix[:, 1:3], axis=1).reshape(-1, 1)
                W_matrix[:, 0] += np.mean(W_matrix[:, 3:7], axis=1)
                W_matrix[:, 3:7] -= np.mean(W_matrix[:, 3:7], axis=1).reshape(-1, 1)
                W_matrix[:, 0] += np.mean(W_matrix[:, 7:11], axis=1)
                W_matrix[:, 7:11] -= np.mean(W_matrix[:, 7:11], axis=1).reshape(-1, 1)
                self.W_l_r_normalized = W_matrix
            else:
                W_matrix = self.W_l_r_normalized

        # Get smallest and largest parameter to equalize y-limits in the plots
        w_min, w_max = np.min(W_matrix[:, 1:]), np.max(W_matrix[:, 1:])

        # Make a plot for every element in beta (#ofplots = #ofparameters)
        fig, axes = plt.subplots(3, 4, figsize=(16, 9))
        axes = axes.flatten()
        titles = [r'$\tilde{\alpha}_{base}$', r'$\tilde{\alpha}_{p}$', r'$\tilde{\alpha}_{up}$',
                  r'$\tilde{\alpha}_{A,l}$', r'$\tilde{\alpha}_{C,l}$', r'$\tilde{\alpha}_{G,l}$',
                  r'$\tilde{\alpha}_{T,l}$', r'$\tilde{\alpha}_{A,r}$', r'$\tilde{\alpha}_{C,r}$',
                  r'$\tilde{\alpha}_{G,r}$', r'$\tilde{\alpha}_{T,r}$']

        # At the third position there is no plot but information
        axes[3].axis('off')
        axes[3].text(0.1, 0.5, self.reg_type + ' reg. = ' + str(self.reg_strength) + '\nmodel type: ' + self.type, fontsize=14)

        for i in range(0, 11):
            ax = axes[i] if i < 3 else axes[i + 1]
            ax.axhline(0, color='black')
            ax.set_title(titles[i], fontsize=10)
            ax.bar(mut_types, W_matrix[:, i].flatten(), align='center', width=0.8,
                   color=[mut_type_cols[i] for i in range(12)])
            if (i + 1) % 4 != 0 and i != 0 and i != 1:
                ax.set_yticklabels([])
            if i > 6:
                ax.set_xlabel('mutation type')
            if i > 0:
                ax.set_ylim(1.05 * w_min, 1.05 * w_max) if y_lims is None else ax.set_ylim(1.05 * y_lims[0], 1.05 * y_lims[1])
        plt.tight_layout()
        plt.show()

        # Plot w for every mutation type (12 plots)
        fig, axes = plt.subplots(3, 4, figsize=(16, 9))
        for i, mut_type in enumerate(mut_types):
            ax = axes[plot_map[mut_type]]
            ax.axhline(0, color='black')
            values = W_matrix[i, 1:]
            ax.bar(titles[1:], values, align='center', width=0.8, color=mut_type_cols[i])
            ax.set_title(mut_type[0] + r'$\rightarrow$' + mut_type[1], fontsize=10)
            if plot_map[mut_type][1] != 0:
                ax.set_yticks([])
            ax.set_ylim(1.05 * w_min, 1.05 * w_max) if y_lims is None else ax.set_ylim(1.05 * y_lims[0],
                                                                                       1.05 * y_lims[1])
        plt.tight_layout()
        plt.show()

    def create_data_matrix(self, mut_counts_df):

        unpaired = mut_counts_df['unpaired'].values
        # TODO: The following assumes currently that the context in the passed dataframe is -2 to +2.
        context_l = mut_counts_df['context'].apply(lambda x: x[1]).values
        context_r = mut_counts_df['context'].apply(lambda x: x[3]).values

        base = np.full(len(unpaired), 1)

        columns = [base]
        if self.type == 'p_up':
            columns += [unpaired]
        elif self.type == 'l_r':
            columns += [unpaired] + self.one_hot_l_r(context_l, context_r)
        elif self.type == 'l_r_st':
            columns += [unpaired] + self.one_hot_l_r_st(context_l, context_r, unpaired)
        elif self.type == 'lr':
            columns += [unpaired] + self.one_hot_lr(context_l, context_r)
        elif self.type == 'lr_pup':
            columns += self.one_hot_lr_pup(context_l, context_r, unpaired)

        X = np.column_stack(columns)

        indices = {}
        present_mut_types = mut_counts_df['nt_mutation'].apply(lambda x: x[0] + x[-1])
        for mut_type in present_mut_types.unique():
            indices[mut_type] = np.where(present_mut_types == mut_type)

        return X, indices

    @staticmethod
    def one_hot_l_r(context_l, context_r):
        sigma = {}
        for nt in ['C', 'G', 'T']:
            sigma[nt + '_l'] = (context_l == nt).astype(int)
            sigma[nt + '_r'] = (context_r == nt).astype(int)

        return [sigma['C_l'], sigma['G_l'], sigma['T_l'],
                sigma['C_r'], sigma['G_r'], sigma['T_r']]

    @staticmethod
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

    @staticmethod
    def one_hot_lr(context_l, context_r):
        sigma = {}
        for nt1 in letters:
            for nt2 in letters:
                sigma[nt1 + nt2] = (context_l + context_r == nt1 + nt2).astype(int)

        return [sigma['AC'], sigma['AG'], sigma['AT'], sigma['CA'], sigma['CC'], sigma['CG'],
                sigma['CT'], sigma['GA'], sigma['GC'], sigma['GG'], sigma['GT'], sigma['TA'],
                sigma['TC'], sigma['TG'], sigma['TT']]

    @staticmethod
    def one_hot_lr_pup(context_l, context_r, unpaired):
        sigma = {}
        for nt1 in letters:
            for nt2 in letters:
                for state in [0, 1]:
                    st = 'p' if state == 0 else 'up'
                    sigma[nt1 + nt2 + '_' + st] = ((context_l + context_r == nt1 + nt2) * (unpaired == state)).astype(
                        int)

        return [sigma['AA_up'], sigma['AC_p'], sigma['AC_up'], sigma['AG_p'], sigma['AG_up'],
                sigma['AT_p'], sigma['AT_up'], sigma['CA_p'], sigma['CA_up'], sigma['CC_p'], sigma['CC_up'],
                sigma['CG_p'], sigma['CG_up'], sigma['CT_p'], sigma['CT_up'], sigma['GA_p'], sigma['GA_up'],
                sigma['GC_p'], sigma['GC_up'], sigma['GG_p'], sigma['GG_up'], sigma['GT_p'], sigma['GT_up'],
                sigma['TA_p'], sigma['TA_up'], sigma['TC_p'], sigma['TC_up'], sigma['TG_p'], sigma['TG_up'],
                sigma['TT_p'], sigma['TT_up']]

    def predict_log_counts(self):

        X, indices = self.create_data_matrix(self.df_test.copy())

        predicted_log_counts = np.zeros(X.shape[0])

        present_mut_types = self.df_test['nt_mutation'].apply(lambda x: x[0] + x[-1])
        for mut_type in present_mut_types.unique():
            indices = np.where(present_mut_types == mut_type)
            predicted_log_counts[indices] = X[indices] @ self.W[mut_type]

        return predicted_log_counts

    def plot_fit_vs_data(self):

        # Prepare figure with the same aspect ratio as Keynote slides
        fig, axes = plt.subplots(3, 4, figsize=(16, 9))

        df_local = self.df_train.copy()

        df_local['context'] = df_local['context'].apply(lambda x: x[1:4])

        # Loop over all 12 mutation types
        for mut_type in mut_types:

            df = df_local[df_local['nt_mutation'].str.match('^' + mut_type[0] + '.*' + mut_type[1] + '$')]

            # Select correct figure for this mutation type
            ax = axes[plot_map[mut_type]]

            # Prepare dictionaries with contexts as keys
            observed_mean_paired, observed_mean_unpaired = {}, {}
            predicted_mean_paired, predicted_mean_unpaired = {}, {}

            # Loop over all 16 contexts
            for context in contexts:

                # Set context
                context = context[0] + mut_type[0] + context[1]

                # Choose sites
                df_paired = df[(df['context'] == context) & (df['unpaired'] == 0)]
                df_unpaired = df[(df['context'] == context) & (df['unpaired'] == 1)]

                # Get mean log counts for every context/pairing state
                observed_mean_paired[context] = np.mean(np.log(df_paired['actual_count'].values + 0.5)) if len(
                    df_paired) > 0 else 0
                observed_mean_unpaired[context] = np.mean(np.log(df_unpaired['actual_count'].values + 0.5)) if len(
                    df_unpaired) > 0 else 0

                nt_l, nt_r = context[0], context[2]

                x_p = np.array([1])
                x_up = np.array([1])
                if self.type == 'p_up':
                    x_p = np.array([1, 0]).astype(int)
                    x_up = np.array([1, 1]).astype(int)
                if self.type == 'l_r':
                    x_p = np.array([1, 0, nt_l == 'C', nt_l == 'G', nt_l == 'T',
                                    nt_r == 'C', nt_r == 'G', nt_r == 'T']).astype(int)
                    x_up = np.array([1, 1, nt_l == 'C', nt_l == 'G', nt_l == 'T',
                                     nt_r == 'C', nt_r == 'G', nt_r == 'T']).astype(int)
                if self.type == 'l_r_st':
                    x_p = np.array([1, 0, 0, nt_l == 'C', 0, nt_l == 'G', 0, nt_l == 'T', 0,
                                    0, nt_r == 'C', 0, nt_r == 'G', 0, nt_r == 'T', 0]).astype(int)
                    x_up = np.array([1, 1, nt_l == 'A', 0, nt_l == 'C', 0, nt_l == 'G', 0, nt_l == 'T',
                                     nt_r == 'A', 0, nt_r == 'C', 0, nt_r == 'G', 0, nt_r == 'T']).astype(int)
                if self.type == 'lr':
                    k = nt_l + nt_r
                    x_p = np.array([1, 0, k == 'AC', k == 'AG', k == 'AT', k == 'CA', k == 'CC',
                                    k == 'CG', k == 'CT', k == 'GA', k == 'GC', k == 'GG', k == 'GT',
                                    k == 'TA', k == 'TC', k == 'TG', k == 'TT']).astype(int)
                    x_up = np.array([1, 1, k == 'AC', k == 'AG', k == 'AT', k == 'CA', k == 'CC',
                                     k == 'CG', k == 'CT', k == 'GA', k == 'GC', k == 'GG', k == 'GT',
                                     k == 'TA', k == 'TC', k == 'TG', k == 'TT']).astype(int)
                if self.type == 'lr_pup':
                    k = nt_l + nt_r
                    x_p = np.array([1, 0, k == 'AC', 0, k == 'AG', 0, k == 'AT', 0, k == 'CA', 0, k == 'CC', 0,
                                    k == 'CG', 0, k == 'CT', 0, k == 'GA', 0, k == 'GC', 0, k == 'GG', 0, k == 'GT',
                                    0, k == 'TA', 0, k == 'TC', 0, k == 'TG', 0, k == 'TT', 0]).astype(int)
                    x_up = np.array([1, k == 'AA', 0, k == 'AC', 0, k == 'AG', 0, k == 'AT', 0,
                                     k == 'CA', 0, k == 'CC', 0, k == 'CG', 0, k == 'CT', 0, k == 'GA',
                                     0, k == 'GC', 0, k == 'GG', 0, k == 'GT', 0, k == 'TA', 0, k == 'TC',
                                     0, k == 'TG', 0, k == 'TT']).astype(int)

                predicted_mean_paired[context] = (self.W[mut_type].T @ x_p)[0]
                predicted_mean_unpaired[context] = (self.W[mut_type].T @ x_up)[0]

            keys = [s[0] + '_' + s[2] for s in predicted_mean_paired.keys()]
            obs_mu_p = observed_mean_paired.values()
            obs_mu_up = observed_mean_unpaired.values()
            pred_mu_p = predicted_mean_paired.values()
            pred_mu_up = predicted_mean_unpaired.values()
            ax.bar(np.arange(16) - 0.2, obs_mu_p, tick_label=keys, width=0.4, color='blue', alpha=0.7,
                   label='paired')
            ax.bar(np.arange(16) + 0.2, obs_mu_up, tick_label=keys, width=0.4, color='red', alpha=0.7,
                   label='unpaired')
            ax.bar(np.arange(16) - 0.2, pred_mu_p, tick_label=keys, width=0.4, color='none', edgecolor='black',
                   alpha=0.7, label='predicted', hatch="//")
            ax.bar(np.arange(16) + 0.2, pred_mu_up, tick_label=keys, width=0.4, color='none', edgecolor='black',
                   alpha=0.7, hatch="//")

            ax.set_xticks(np.arange(16), keys, rotation='vertical', fontsize=10)
            ax.set_title(rf"{mut_type[0]}$\rightarrow${mut_type[1]}")
            if plot_map[mut_type] == (0, 0):
                ax.legend()
            if plot_map[mut_type][1] == 0:
                ax.set_ylabel('mean log(n + 0.5)')

        plt.suptitle(f'model type: {self.type}, reg. type: {self.reg_type}, reg. strength: {self.reg_strength}')
        plt.tight_layout()
        plt.savefig(f'results/{CLADE}/{self.type}/predicted_vs_observed_means.png')
        plt.show()


def plot_mse(mean_sq_err_dic):

    model_types = list(mean_sq_err_dic.keys())

    # Create one trace for every mutation type
    traces = {mut_type: [] for mut_type in mean_sq_err_dic[model_types[0]].keys()}

    for model_type in model_types:  # Loop over model types
        for mut_type in traces.keys():  # Loop over mutation types
            traces[mut_type].append(mean_sq_err_dic[model_type][mut_type])

    plt.figure(figsize=(8, 4.5))

    mse_mean = np.zeros(len(traces['AC']))

    # Add traces to plot
    for i, [mut_type, mean_sq_errs] in enumerate(sorted(traces.items())):
        mse_mean += mean_sq_errs
        plt.plot(model_types, mean_sq_errs, 'o-', color=mut_type_cols[i % len(mut_type_cols)],
                 label=rf"{mut_type[0]}$\rightarrow${mut_type[1]}")

    plt.plot(model_types, mse_mean / 12, 'o-', linewidth=2, markersize=8, label='mean', color='black')

    plt.xlabel('type of general linear model')
    plt.ylabel('mean squared error')
    plt.title(f"mean sq. err. on training data ({CLADE}, regularization: {REGULARIZATION})")
    plt.legend()
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'results/{CLADE}/mse_{CLADE}')
    plt.show()


if __name__ == '__main__':

    CLADE = 'curated'

    full_df = load_mut_counts(clade=CLADE, mut_types='synonymous', include_noncoding=False, include_tolerant_orfs=False,
                              remove_orf9b=False)

    model_versions = ['base', 'p_up', 'l_r', 'l_r_st', 'lr', 'lr_pup'][2:3]

    REGULARIZATION = [('l1', 0.01), ('l2', 0.1)][1]

    mean_squared_errs = {}

    for model_version in model_versions:

        model = GeneralLinearModel(type=model_version, training_data=full_df.copy(), regularization=REGULARIZATION)

        model.train()

        mean_squared_errs[model_version] = model.mean_sq_errs

    #plot_mse(mean_squared_errs)
