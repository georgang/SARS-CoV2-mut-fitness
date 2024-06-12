import os
import numpy as np
from Bio import SeqIO
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})


"""This script saves a primitive alignment view in 'results/alignment_view' and prints information about the 
alignment of the bat sequences to Wuhan-Hu-1."""

# TODO: Organize script better using functions


# Create directory to store alignment views
figures_dir = "results/alignment_view"
os.makedirs(figures_dir, exist_ok=True)

# Import original bat sequences downloaded from GISAID (shorter than 29'903 nts)
original_seqs = SeqIO.to_dict(SeqIO.parse('data/2024-05-17_bat_sequences.fasta', "fasta"))

# Import bat sequences aligned to Wuhan-Hu-1 by Nextclade (29'903 nts)
aligned_seqs = SeqIO.to_dict(SeqIO.parse('results/alignment.fasta', "fasta"))

# Extract Wuhan-Hu-1 as reference sequence
wuhan_hu_1 = np.array(list(aligned_seqs['MN908947']))

# Prepare array for the visualization of the gap positions
gap_pos = np.zeros(29903)

# Load boundaries of non-recombinant regions
breakpoints = np.loadtxt('results/breakpoints.txt')

# Load names and boundaries of genes
# TODO: Check this file again
genes = []
with open('/Users/georgangehrn/Desktop/SARS2-mut-fitness-corr/human_data/genes.txt', 'r') as file:
    for line in file:
        items = line.strip().split(',')
        genes.append((int(items[0]), int(items[1]), items[2]))


def plot_sequence_differences(reference, aligned, id_ref, id_al, regions=genes):
    """
    Parameters:
    - reference (str): reference nucleotide sequence (containing only A, C, G, T)
    - aligned (str): aligned nucleotide sequence (containing A, C, G, T, -)
    """

    if len(reference) != len(aligned):
        raise ValueError("The sequences must be of the same length.")

    segments = [(0, 10000), (10000, 20000), (20000, 30000)]

    colors = {
        'mismatch': 'red',
        'gap': 'black',
        'region': 'gray'
    }

    fig, axes = plt.subplots(3, 1, figsize=(15, 6), sharex=False)

    n_of_gaps = 0
    n_of_mismatches = 0

    for idx, (start, end) in enumerate(segments):

        x_positions = []
        heights = []
        bar_colors = []

        if idx == 0:
            x_positions.append(0)
            heights.append(0.0)
            bar_colors.append('green')

        for j in range(start, min(end, len(reference))):
            if reference[j] != aligned[j]:
                x_positions.append(j + 1)
                heights.append(1.0)
                if aligned[j] == '-':
                    bar_colors.append(colors['gap'])
                    n_of_gaps += 1
                else:
                    bar_colors.append(colors['mismatch'])
                    n_of_mismatches += 1

        if idx == 2:
            x_positions.append(30000)
            heights.append(0.0)
            bar_colors.append('green')

        # width 4 was the empirically found visibility limit
        # TODO: Maybe it would be better to average over several positions
        axes[idx].bar(x_positions, heights, color=bar_colors, width=4, align='center', alpha=0.5)

        for k, (region_start, region_end, region_name) in enumerate(regions):
            if start <= region_start <= end or start <= region_end <= end:
                offset = 0.1 * (1 - (k % 2))
                axes[idx].axvspan(xmin=max(start, region_start), xmax=min(end, region_end), ymin=0.8 + offset, ymax=0.9 + offset, color=colors['region'], alpha=1.0, transform=axes[idx].get_xaxis_transform())
                if region_start >= start:
                    axes[idx].text(max(start, region_start), 0.8 + offset, region_name, color='black', ha='left', fontsize=10)
        axes[idx].set_ylim(0, 1)
        axes[idx].set_yticks([])

        for bp in breakpoints[1:-1]:
            if start <= bp < end:
                axes[idx].axvline(bp, color='blue', linestyle='--', alpha=1, ymin=-0.1, ymax=1.1)

    axes[0].text(100, 0.6, f'gaps (black): {n_of_gaps}', fontsize=12, color='black')
    axes[0].text(100, 0.4, f'mismatches (red): {n_of_mismatches}', fontsize=12, color='red')
    axes[0].text(100, 0.2, f'recombination breakpoints', fontsize=12, color='blue')

    axes[-1].set_xlabel('Nucleotide Position')
    fig.suptitle(f'{id_al} \nvs. {id_ref}')

    seq_id = id_al.replace('/', '_')
    figure_name = os.path.join(figures_dir, f'{seq_id}.png')
    plt.tight_layout()
    plt.savefig(figure_name)
    plt.close()


# Loop over all bat isolates and visualise agreement with Wuhan-Hu-1
for seq_id, seq in aligned_seqs.items():

    # Exclude Wuhan-Hu-1
    if seq_id != 'MN908947':

        print(f"\nProcessing sequence with ID {seq_id}")

        # Original sequence downloaded from GISAID
        original_seq = np.array(list(original_seqs[seq_id]))

        # Count number of letters in the original sequence that are not A, C, G, or T
        n_of_errors = np.count_nonzero(1 - np.isin(original_seq, ['A', 'C', 'G', 'T']))

        # Aligned sequence as array
        aligned_seq = np.array(list(seq))

        # Print number of gaps in aligned sequence
        n_of_gaps = np.count_nonzero(aligned_seq == '-')
        print(f'- Number of gaps (\'-\') in aligned sequence: {n_of_gaps}')

        # Print number of differences between aligned sequence and Wuhan-Hu-1 that are not gaps
        differences = np.count_nonzero(aligned_seq != wuhan_hu_1) - n_of_gaps
        print(f'- Number of differences compared to Wuhan-Hu-1 (gaps excluded): {differences}')

        # Plot differences between Wuhan-Hu-1 and aligned bat sequence
        plot_sequence_differences(wuhan_hu_1, aligned_seq, id_ref='Wuhan-Hu-1', id_al=seq_id)

        # Remove gaps
        mask = aligned_seq != '-'
        gap_pos += 1 - mask
        aligned_seq_wo_gaps = aligned_seq[mask]

        if len(aligned_seq_wo_gaps) == len(original_seq):
            if np.count_nonzero(aligned_seq_wo_gaps != original_seq) == 0:
                print('- Aligned sequence without gaps is equal to the original sequence.')
            else:
                errors = np.count_nonzero(aligned_seq_wo_gaps != original_seq)
                print(
                    f'- Aligned sequence without gaps has the same length as the original sequence, but {errors} errors')
        else:
            print('- Aligned sequence without gaps has a different length than the original sequence:')
            pos = 0
            same = aligned_seq_wo_gaps[pos] == original_seq[pos]
            while same:
                pos += 1
                same = aligned_seq_wo_gaps[pos] == original_seq[pos]
            print(f'  The first error occurs at position {pos + 1} (without gaps)')
            print(f'  -1 to +10 context of this position is:')
            print(f'  aligned:  {aligned_seq_wo_gaps[pos - 1:pos + 10]}')
            print(f'  original: {original_seq[pos - 1:pos + 10]}')


# Visualize position of gaps
gap_pos = np.array_split(gap_pos, 3)
site_num = np.array_split(np.arange(29903), 3)
fig, axes = plt.subplots(3, 1, figsize=(20, 10))
for i, ax in enumerate(axes.flat):
    ax.bar(site_num[i], gap_pos[i], color='black', width=1)
    if i == 2:
        ax.set_xlabel('Nucleotide Position')
    ax.set_ylabel('# of seqs. with gap')
    ax.set_yticks(np.arange(len(aligned_seqs) + 1)[::4])
    ax.set_ylim((0, len(aligned_seqs) + 1))
plt.suptitle(f'Gaps in the Nextclade alignment of {len(aligned_seqs) - 1} bat sequences to Wuhan-Hu-1')
plt.tight_layout()
gaps_figure_name = os.path.join(figures_dir, 'gaps_in_alignment.png')
plt.savefig(gaps_figure_name)
plt.close()
plt.show()
