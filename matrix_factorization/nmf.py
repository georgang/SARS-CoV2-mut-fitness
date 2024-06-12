import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error

plt.rcParams.update({'font.size': 12})

row_keys = ['AAA_C_p', 'AAC_C_p', 'AAG_C_p', 'AAT_C_p', 'CAA_C_p', 'CAC_C_p', 'CAG_C_p', 'CAT_C_p', 'GAA_C_p',
            'GAC_C_p', 'GAG_C_p', 'GAT_C_p', 'TAA_C_p', 'TAC_C_p', 'TAG_C_p', 'TAT_C_p', 'AAA_C_up', 'AAC_C_up',
            'AAG_C_up', 'AAT_C_up', 'CAA_C_up', 'CAC_C_up', 'CAG_C_up', 'CAT_C_up', 'GAA_C_up', 'GAC_C_up', 'GAG_C_up',
            'GAT_C_up', 'TAA_C_up', 'TAC_C_up', 'TAG_C_up', 'TAT_C_up', 'AAA_G_p', 'AAC_G_p', 'AAG_G_p', 'AAT_G_p',
            'CAA_G_p', 'CAC_G_p', 'CAG_G_p', 'CAT_G_p', 'GAA_G_p', 'GAC_G_p', 'GAG_G_p', 'GAT_G_p', 'TAA_G_p',
            'TAC_G_p', 'TAG_G_p', 'TAT_G_p', 'AAA_G_up', 'AAC_G_up', 'AAG_G_up', 'AAT_G_up', 'CAA_G_up', 'CAC_G_up',
            'CAG_G_up', 'CAT_G_up', 'GAA_G_up', 'GAC_G_up', 'GAG_G_up', 'GAT_G_up', 'TAA_G_up', 'TAC_G_up', 'TAG_G_up',
            'TAT_G_up', 'AAA_T_p', 'AAC_T_p', 'AAG_T_p', 'AAT_T_p', 'CAA_T_p', 'CAC_T_p', 'CAG_T_p', 'CAT_T_p',
            'GAA_T_p', 'GAC_T_p', 'GAG_T_p', 'GAT_T_p', 'TAA_T_p', 'TAC_T_p', 'TAG_T_p', 'TAT_T_p', 'AAA_T_up',
            'AAC_T_up', 'AAG_T_up', 'AAT_T_up', 'CAA_T_up', 'CAC_T_up', 'CAG_T_up', 'CAT_T_up', 'GAA_T_up', 'GAC_T_up',
            'GAG_T_up', 'GAT_T_up', 'TAA_T_up', 'TAC_T_up', 'TAG_T_up', 'TAT_T_up', 'ACA_A_p', 'ACC_A_p', 'ACG_A_p',
            'ACT_A_p', 'CCA_A_p', 'CCC_A_p', 'CCG_A_p', 'CCT_A_p', 'GCA_A_p', 'GCC_A_p', 'GCG_A_p', 'GCT_A_p',
            'TCA_A_p', 'TCC_A_p', 'TCG_A_p', 'TCT_A_p', 'ACA_A_up', 'ACC_A_up', 'ACG_A_up', 'ACT_A_up', 'CCA_A_up',
            'CCC_A_up', 'CCG_A_up', 'CCT_A_up', 'GCA_A_up', 'GCC_A_up', 'GCG_A_up', 'GCT_A_up', 'TCA_A_up', 'TCC_A_up',
            'TCG_A_up', 'TCT_A_up', 'ACA_G_p', 'ACC_G_p', 'ACG_G_p', 'ACT_G_p', 'CCA_G_p', 'CCC_G_p', 'CCG_G_p',
            'CCT_G_p', 'GCA_G_p', 'GCC_G_p', 'GCG_G_p', 'GCT_G_p', 'TCA_G_p', 'TCC_G_p', 'TCG_G_p', 'TCT_G_p',
            'ACA_G_up', 'ACC_G_up', 'ACG_G_up', 'ACT_G_up', 'CCA_G_up', 'CCC_G_up', 'CCG_G_up', 'CCT_G_up', 'GCA_G_up',
            'GCC_G_up', 'GCG_G_up', 'GCT_G_up', 'TCA_G_up', 'TCC_G_up', 'TCG_G_up', 'TCT_G_up', 'ACA_T_p', 'ACC_T_p',
            'ACG_T_p', 'ACT_T_p', 'CCA_T_p', 'CCC_T_p', 'CCG_T_p', 'CCT_T_p', 'GCA_T_p', 'GCC_T_p', 'GCG_T_p',
            'GCT_T_p', 'TCA_T_p', 'TCC_T_p', 'TCG_T_p', 'TCT_T_p', 'ACA_T_up', 'ACC_T_up', 'ACG_T_up', 'ACT_T_up',
            'CCA_T_up', 'CCC_T_up', 'CCG_T_up', 'CCT_T_up', 'GCA_T_up', 'GCC_T_up', 'GCG_T_up', 'GCT_T_up', 'TCA_T_up',
            'TCC_T_up', 'TCG_T_up', 'TCT_T_up', 'AGA_A_p', 'AGC_A_p', 'AGG_A_p', 'AGT_A_p', 'CGA_A_p', 'CGC_A_p',
            'CGG_A_p', 'CGT_A_p', 'GGA_A_p', 'GGC_A_p', 'GGG_A_p', 'GGT_A_p', 'TGA_A_p', 'TGC_A_p', 'TGG_A_p',
            'TGT_A_p', 'AGA_A_up', 'AGC_A_up', 'AGG_A_up', 'AGT_A_up', 'CGA_A_up', 'CGC_A_up', 'CGG_A_up', 'CGT_A_up',
            'GGA_A_up', 'GGC_A_up', 'GGG_A_up', 'GGT_A_up', 'TGA_A_up', 'TGC_A_up', 'TGG_A_up', 'TGT_A_up', 'AGA_C_p',
            'AGC_C_p', 'AGG_C_p', 'AGT_C_p', 'CGA_C_p', 'CGC_C_p', 'CGG_C_p', 'CGT_C_p', 'GGA_C_p', 'GGC_C_p',
            'GGG_C_p', 'GGT_C_p', 'TGA_C_p', 'TGC_C_p', 'TGG_C_p', 'TGT_C_p', 'AGA_C_up', 'AGC_C_up', 'AGG_C_up',
            'AGT_C_up', 'CGA_C_up', 'CGC_C_up', 'CGG_C_up', 'CGT_C_up', 'GGA_C_up', 'GGC_C_up', 'GGG_C_up', 'GGT_C_up',
            'TGA_C_up', 'TGC_C_up', 'TGG_C_up', 'TGT_C_up', 'AGA_T_p', 'AGC_T_p', 'AGG_T_p', 'AGT_T_p', 'CGA_T_p',
            'CGC_T_p', 'CGG_T_p', 'CGT_T_p', 'GGA_T_p', 'GGC_T_p', 'GGG_T_p', 'GGT_T_p', 'TGA_T_p', 'TGC_T_p',
            'TGG_T_p', 'TGT_T_p', 'AGA_T_up', 'AGC_T_up', 'AGG_T_up', 'AGT_T_up', 'CGA_T_up', 'CGC_T_up', 'CGG_T_up',
            'CGT_T_up', 'GGA_T_up', 'GGC_T_up', 'GGG_T_up', 'GGT_T_up', 'TGA_T_up', 'TGC_T_up', 'TGG_T_up', 'TGT_T_up',
            'ATA_A_p', 'ATC_A_p', 'ATG_A_p', 'ATT_A_p', 'CTA_A_p', 'CTC_A_p', 'CTG_A_p', 'CTT_A_p', 'GTA_A_p',
            'GTC_A_p', 'GTG_A_p', 'GTT_A_p', 'TTA_A_p', 'TTC_A_p', 'TTG_A_p', 'TTT_A_p', 'ATA_A_up', 'ATC_A_up',
            'ATG_A_up', 'ATT_A_up', 'CTA_A_up', 'CTC_A_up', 'CTG_A_up', 'CTT_A_up', 'GTA_A_up', 'GTC_A_up', 'GTG_A_up',
            'GTT_A_up', 'TTA_A_up', 'TTC_A_up', 'TTG_A_up', 'TTT_A_up', 'ATA_C_p', 'ATC_C_p', 'ATG_C_p', 'ATT_C_p',
            'CTA_C_p', 'CTC_C_p', 'CTG_C_p', 'CTT_C_p', 'GTA_C_p', 'GTC_C_p', 'GTG_C_p', 'GTT_C_p', 'TTA_C_p',
            'TTC_C_p', 'TTG_C_p', 'TTT_C_p', 'ATA_C_up', 'ATC_C_up', 'ATG_C_up', 'ATT_C_up', 'CTA_C_up', 'CTC_C_up',
            'CTG_C_up', 'CTT_C_up', 'GTA_C_up', 'GTC_C_up', 'GTG_C_up', 'GTT_C_up', 'TTA_C_up', 'TTC_C_up', 'TTG_C_up',
            'TTT_C_up', 'ATA_G_p', 'ATC_G_p', 'ATG_G_p', 'ATT_G_p', 'CTA_G_p', 'CTC_G_p', 'CTG_G_p', 'CTT_G_p',
            'GTA_G_p', 'GTC_G_p', 'GTG_G_p', 'GTT_G_p', 'TTA_G_p', 'TTC_G_p', 'TTG_G_p', 'TTT_G_p', 'ATA_G_up',
            'ATC_G_up', 'ATG_G_up', 'ATT_G_up', 'CTA_G_up', 'CTC_G_up', 'CTG_G_up', 'CTT_G_up', 'GTA_G_up', 'GTC_G_up',
            'GTG_G_up', 'GTT_G_up', 'TTA_G_up', 'TTC_G_up', 'TTG_G_up', 'TTT_G_up']

host_keys = ['21J p', '21K p', 'bats p', 'deer p',
             '21J up', '21K up', 'bats up', 'deer up']

x_keys = [item[:-4] for item in row_keys if item.endswith('_p')]

mt_type_col = np.repeat(['blue', 'orange', 'green', 'red', 'purple', 'brown',
                         'pink', 'gray', 'cyan', 'magenta', 'lime', 'teal'], 16)

# Load files with calculated rates
rates_files = ['rates_21J.csv', 'rates_21K.csv', 'rates_bats.csv', 'rates_deer.csv']

# Create data matrix X of shape (192, 8) with a column for every combination of host and pairing state
data_frames = [pd.read_csv(file, usecols=[1], header=None).to_numpy().reshape((12, 2, 4, 4)) for file in rates_files]
data_paired = [df[:, 0, :, :].flatten() for df in data_frames]
data_unpaired = [df[:, 1, :, :].flatten() for df in data_frames]
X = np.vstack((data_paired, data_unpaired))
X = (X.T / np.sum(X.T, axis=0)).T  # Normalize such that every column adds up to 1
max_rate = 1.1 * np.max(X)

# Visualise mutation spectra
n = X.shape[0]
fig, axes = plt.subplots(int(n / 2), 2, figsize=(24, 3 * n / 2), dpi=200)
for i in range(n):
    ax = axes[i % 4, int(i > 3)]
    ax.bar(np.arange(192), X[i, :], color=[mt_type_col[j] for j in range(192)])
    ax.set_ylim((0, max_rate))
    if i < 4:
        ax.set_ylabel('relative rate')
    if i % 4 == 3:
        ax.set_xlabel('context')
    ax.set_xticks(np.arange(192), x_keys, rotation='vertical', fontsize=5)
    ax.set_title(host_keys[i])
fig.suptitle('Actual mutation spectra', fontsize=16)
plt.tight_layout()
plt.show()


def calculate_reconstruction_error(data, n_components):
    # Initialise NMF
    nmf_model = NMF(n_components=n_components, init='random', random_state=42)

    # Perform matrix factorization
    W = nmf_model.fit_transform(data)
    H = nmf_model.components_

    # Calculate reconstruction error
    data_reconstructed = np.dot(W, H)
    error = mean_squared_error(data, data_reconstructed)

    # Visualise predicted mutation spectra
    n = data_reconstructed.shape[0]
    fig, axes = plt.subplots(int(n / 2), 2, figsize=(24, 3 * n / 2), dpi=200)
    for i in range(n):
        ax = axes[i % 4, int(i > 3)]
        ax.bar(np.arange(192), data_reconstructed[i, :], color=[mt_type_col[j] for j in range(192)])
        ax.set_ylim((0, max_rate))
        if i < 4:
            ax.set_ylabel('relative rate')
        if i % 4 == 3:
            ax.set_xlabel('context')
        ax.set_xticks(np.arange(192), x_keys, rotation='vertical', fontsize=5)
        ax.set_title(host_keys[i])
    fig.suptitle('Predicted mutation spectra (p = ' + str(n_components) + ')', fontsize=16)
    plt.tight_layout()
    plt.show()

    # Normalize matrices
    H = (H.T / np.sum(H, axis=1)).T
    W = W * np.sum(H, axis=1)

    # Plot learned signatures
    fig, axes = plt.subplots(n_components, 1, figsize=(16, 3 * n_components), dpi=200)
    for i in range(n_components):
        ax = (axes if n_components == 1 else axes[i])
        ax.bar(np.arange(192), H[i, :], color=[mt_type_col[j] for j in range(192)])
        ax.set_ylim((0, 1.1 * np.max(H)))
        ax.set_ylabel('h')
        if i == n_components - 1:
            ax.set_xlabel('context')
        ax.set_xticks(np.arange(192), x_keys, rotation='vertical', fontsize=5)
        ax.set_title(f'signature {i + 1}')
    fig.suptitle('Learned signatures', fontsize=16)
    plt.tight_layout()
    plt.show()

    # Visualise the shares of each component in the different environments
    plt.imshow(W.T, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('W')
    plt.xticks(np.arange(8), host_keys, rotation='vertical')
    plt.yticks(np.arange(n_components), [str(i) for i in np.arange(1, n_components + 1)])
    plt.ylabel('signature')
    plt.show()

    # Visualise dot product for every host
    plt.imshow(np.diagonal(data @ data_reconstructed.T).reshape(1, -1), cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('dot product')
    plt.xticks(np.arange(8), host_keys, rotation='vertical')
    plt.yticks([])
    plt.show()

    return error


# Sweep over a range of component numbers
errors = []
component_range = range(1, 9)
for n in component_range:
    error = calculate_reconstruction_error(X, n)
    errors.append((n, error))

# Plot the reconstruction error versus the number of components
components, reconstruction_errors = zip(*errors)
plt.figure(figsize=(10, 6))
plt.plot(components, reconstruction_errors, marker='o', linestyle='--', color='b')
plt.xlabel('Number of NMF Components')
plt.ylabel('Reconstruction Error (Mean Squared Error)')
plt.xticks(components)
plt.grid(True)
plt.show()
