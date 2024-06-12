import argparse
from Bio import Phylo
import matplotlib.pyplot as plt

# TODO: Remove __name__ condition

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=str, required=True, help="Input tree file")
    parser.add_argument("--output", type=str, required=True, help="Output tree file")
    parser.add_argument("--output-plot", type=str, required=True, help="Output tree figure")
    args = parser.parse_args()

    # Read in tree of non-recombinant region inferred by augur/IQ-TREE
    T = Phylo.read(args.tree, 'newick')

    # Root it at the midpoint of the two most distant taxa and save it again in newick format
    T.root_at_midpoint()

    # Add name to internal nodes that will also be adapted by augur ancestral
    for ni, n in enumerate(T.get_nonterminals()):
        n.name = f"NODE_{ni:04d}"
    Phylo.write(T, args.output, 'newick')

    # Store tree as image
    Phylo.draw(T, do_show=False)
    plt.tight_layout()
    plt.savefig(args.output_plot)
    plt.close()
