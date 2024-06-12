from Bio import Phylo
import matplotlib.pyplot as plt


'''This script visualises the phylogenetic tree of every non-recombinant region (NRR) before and after rooting.'''


# Loop through all non-recombinant regions
for nrr in range(4, 5):

    # Read in unrooted tree
    T = Phylo.read(f'/Users/georgangehrn/Desktop/SARS2-mut-fitness-corr/bat_data/results/trees/{nrr:02d}.nwk', 'newick')

    # Label leaves and nodes
    for i, n in enumerate(T.get_terminals()):
        n.name = f"leaf_{i}"

    for i, n in enumerate(T.get_nonterminals()):
        n.name = f"node_{i}"

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Plot unrooted tree on the first subplot
    Phylo.draw(T, do_show=False, axes=axes[0])
    axes[0].set_title(f'NRR {nrr}, unrooted')

    # Root the tree
    T.root_at_midpoint()

    for i, n in enumerate(T.get_nonterminals()):
        if not n.name:
            n.name = "root"

    # Plot rooted tree on the second subplot
    Phylo.draw(T, do_show=False, axes=axes[1])
    axes[1].set_title(f'NRR {nrr}, rooted')

    plt.show()


