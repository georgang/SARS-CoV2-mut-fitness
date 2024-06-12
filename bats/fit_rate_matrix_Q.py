import json
import numpy as np
import pandas as pd
from Bio import Phylo
from scipy.linalg import expm
from scipy.optimize import minimize
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 12})

# Set a seed for reproducibility
np.random.seed(42)

# Set prior probabilities for A, C, G, T
Pi = np.array([0.25, 0.25, 0.25, 0.25])

# Prepare dictionary for the conversion of nucleotide letter to 4-dimensional vector
profile_maps = {
    'nuc_nogap': {
        'A': np.array([1, 0, 0, 0], dtype='float'),
        'C': np.array([0, 1, 0, 0], dtype='float'),
        'G': np.array([0, 0, 1, 0], dtype='float'),
        'T': np.array([0, 0, 0, 1], dtype='float'),
        '-': np.array([0.25, 0.25, 0.25, 0.25], dtype='float')
    }
}

# Load breakpoints of non-recombinant regions
breakpoints = np.loadtxt('results/breakpoints.txt').astype(int)

# Prepare masks for four-fold degenerate un-/paired sites
# TODO: The ffd mask is taken from 21J. Create one for bats. Can I take the reference sequences of the NRRs?
ffd_mask = np.loadtxt('results/masks/ffd_mask.csv').astype(bool)
unpaired_mask = np.loadtxt('results/masks/unpaired_mask.csv').astype(bool)
paired_mask = (1 - unpaired_mask).astype(bool)
masks = {'paired': ffd_mask * paired_mask, 'unpaired': ffd_mask * unpaired_mask}


def seq2prof(seq, profile_map):
    return np.array([profile_map[k] for k in seq])


def propagate_profile(Q, profile, branch_length):
    Qt = Q * branch_length
    exp_Qt = expm(Qt)
    res = exp_Qt @ profile  # (4, 4) * (4, length of sequence)
    return res


def neg_log_lh(qAC, qAG, qAT, qCA, qCG, qCT, qGA, qGC, qGT, qTA, qTC, qTG, tree, seqs, mask):
    # Define rate matrix such that rows sum to 0
    Q = np.array([[-(qAC ** 2 + qAG ** 2 + qAT ** 2), qAC ** 2, qAG ** 2, qAT ** 2],
                  [qCA ** 2, -(qCA ** 2 + qCG ** 2 + qCT ** 2), qCG ** 2, qCT ** 2],
                  [qGA ** 2, qGC ** 2, -(qGA ** 2 + qGC ** 2 + qGT ** 2), qGT ** 2],
                  [qTA ** 2, qTC ** 2, qTG ** 2, -(qTA ** 2 + qTC ** 2 + qTG ** 2)]])

    # Propagate leaves of tree
    for leaf in tree.get_terminals():
        leaf.seq = seqs[leaf.name]
        leaf.profile = seq2prof(leaf.seq, profile_map=profile_maps['nuc_nogap']).T[:, mask]
        leaf.msg_to_parent = propagate_profile(Q, leaf.profile, leaf.branch_length)

    # Propagate inner nodes of tree
    for n in tree.get_nonterminals(order='postorder'):
        n.profile = np.prod([c.msg_to_parent for c in n.clades], axis=0)
        n.msg_to_parent = propagate_profile(Q, n.profile, n.branch_length)

    # Final step (cf. Felsenstein, 1981)
    tree.root.msg_from_parent = Pi
    tree.root.marginal_LH = tree.root.msg_from_parent.reshape(-1, 1) * tree.root.profile
    nll = - np.sum(np.log(tree.root.marginal_LH.sum(axis=0)))

    return nll


def wrapper(params, full_mask):
    qAC, qAG, qAT, qCA, qCG, qCT, qGA, qGC, qGT, qTA, qTC, qTG = params

    total_nll = 0

    for chunk in range(27):
        tree = Phylo.read(f'results/trees/refined_{chunk:02d}.nwk', 'newick')

        chunk_mask = full_mask[breakpoints[chunk]:breakpoints[chunk + 1]]

        with open(f'results/mutations/{chunk:02d}.json', 'r') as f:
            nodes = json.load(f)['nodes']
            seqs = {node: nodes[node]['sequence'] for node in list(nodes.keys())}

        total_nll += neg_log_lh(qAC, qAG, qAT, qCA, qCG, qCT, qGA, qGC, qGT, qTA, qTC, qTG, tree, seqs, chunk_mask)

    return total_nll


def callback(xk):
    global iteration_count
    iteration_count += 1
    nll = wrapper(xk, masks[state])
    nll_values.append(nll)
    print(f"Iteration {iteration_count}: Parameters = {xk}, NLL = {nll}")


def plot_and_save_rates(rates):
    # Extract rates
    unpaired = rates['unpaired']
    paired = rates['paired']

    # Normalize rates
    rates_p = paired / (np.sum(unpaired) + np.sum(paired))
    rates_up = unpaired / (np.sum(unpaired) + np.sum(paired))

    # Plot normalized rates
    plt.figure()
    x_ticks = np.arange(12)
    width = 0.25
    plt.bar(x_ticks + width, rates_p, align='center', color='blue', label='paired', alpha=0.6, width=width)
    plt.bar(x_ticks, rates_up, align='center', color='red', label='unpaired', alpha=0.6, width=width)
    plt.bar(x_ticks - width, rates_p + rates_up, align='center', color='black', label='total', alpha=0.6, width=width)
    plt.xticks(x_ticks,
               [r'A$\rightarrow$C', r'A$\rightarrow$G', r'A$\rightarrow$T', r'C$\rightarrow$A', r'C$\rightarrow$G',
                r'C$\rightarrow$T', r'G$\rightarrow$A', r'G$\rightarrow$C', r'G$\rightarrow$T', r'T$\rightarrow$A',
                r'T$\rightarrow$C', r'T$\rightarrow$G'],
               rotation=45)
    plt.xlabel('mutation type', fontsize=12)
    plt.ylabel('normalized mutation rate', fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.grid(axis='y')
    plt.savefig('results/mut_rates/bat_rates.png')
    plt.show()

    # Save rates
    names_p = ["A_C_p", "A_G_p", "A_T_p", "C_A_p", "C_G_p", "C_T_p", "G_A_p", "G_C_p", "G_T_p", "T_A_p", "T_C_p",
               "T_G_p"]
    names_up = ["A_C_up", "A_G_up", "A_T_up", "C_A_up", "C_G_up", "C_T_up", "G_A_up", "G_C_up", "G_T_up", "T_A_up",
                "T_C_up", "T_G_up"]

    # Combine the arrays
    rates_all = list(rates_p) + list(rates_up)
    names_all = names_p + names_up

    # Create the DataFrame
    df = pd.DataFrame({
        'mutation': names_all,
        'relative_rate': rates_all
    })

    df.to_csv('results/mut_rates/bat_rates.csv', index=False, header=True)


def plot_nll(values, pairing_state):
    plt.plot(range(1, len(values) + 1), values, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Negative Log Likelihood')
    plt.title(pairing_state)
    plt.grid()
    plt.tight_layout()
    plt.savefig('results/mut_rates/nll_' + pairing_state + '.png')
    plt.show()


if __name__ == '__main__':

    # Optimize rates for paired and unpaired sites separately
    optimized_rates = {'paired': np.ones(12), 'unpaired': np.ones(12)}
    for state in ['paired', 'unpaired']:
        nll_values = []
        iteration_count = 0
        initial_guess = np.full(12, 0.1)
        mask = masks[state]
        result = minimize(fun=wrapper, x0=initial_guess, args=(mask,), method='L-BFGS-B', callback=callback)
        optimized_rates[state] = result.x ** 2

        plot_nll(nll_values, state)  # TODO: Combine 'paired' and 'unpaired' in one figure

    # Plot and save optimized rates
    plot_and_save_rates(optimized_rates)
