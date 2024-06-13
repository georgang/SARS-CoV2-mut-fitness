import numpy as np
import pandas as pd
from helper import mut_types, plot_map
from fit_general_linear_models import GeneralLinearModel


if __name__ == '__main__':

    '''I am not saving any of the figures here, but put them directly into the slides.'''

    # Load parameters of 'l_r' general linear model for two clades
    glm_21J = GeneralLinearModel('l_r', regularization=('l2', 0.1))

    glm_21J.W = pd.read_csv(f"/Users/georgangehrn/Desktop/SARS-CoV2-mut-fitness/general_linear_models/results/21J"
                            f"/l_r/learned_params.csv").to_dict(orient='list')

    glm_21K = GeneralLinearModel('l_r', regularization=('l2', 0.1))

    glm_21K.W = pd.read_csv(f"/Users/georgangehrn/Desktop/SARS-CoV2-mut-fitness/general_linear_models/results/21K"
                            f"/l_r/learned_params.csv").to_dict(orient='list')

    # Plot parameters of both models
    glm_21J.plot_l_r_params(normalize=False, y_lims=None)
    W_help_1 = np.vstack([glm_21J.W[mut_type] for mut_type in mut_types])
    glm_21K.plot_l_r_params(normalize=False, y_lims=(np.min(W_help_1[:, 1:]), np.max(W_help_1[:, 1:])))

    # Plot difference between the two clades
    glm_dummy = GeneralLinearModel('l_r', regularization=('l2', 0.1))
    glm_dummy.W = {mut_type: np.array(glm_21J.W[mut_type]) - np.array(glm_21K.W[mut_type]) for mut_type in mut_types}
    glm_dummy.plot_l_r_params(normalize=False, y_lims=(np.min(W_help_1[:, 1:]), np.max(W_help_1[:, 1:])))

    # Plot normalized parameters of both models
    glm_21J.plot_l_r_params(normalize=True, y_lims=(np.min(W_help_1[:, 1:]), np.max(W_help_1[:, 1:])))
    glm_21K.plot_l_r_params(normalize=True, y_lims=(np.min(W_help_1[:, 1:]), np.max(W_help_1[:, 1:])))

    # Plot difference between the two clades
    glm_dummy = GeneralLinearModel('l_r', regularization=('l2', 0.1))
    glm_dummy.W = {mut_type: np.array(glm_21J.W[mut_type]) - np.array(glm_21K.W[mut_type]) for mut_type in mut_types}
    glm_dummy.W_l_r_normalized = glm_21J.W_l_r_normalized - glm_21K.W_l_r_normalized
    glm_dummy.plot_l_r_params(normalize=True, y_lims=(np.min(W_help_1[:, 1:]), np.max(W_help_1[:, 1:])))
