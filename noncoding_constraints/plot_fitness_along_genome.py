import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from Bio import SeqIO

from helper import load_mut_counts, gene_boundaries
from general_linear_models.fit_general_linear_models import add_predictions
import matplotlib.patches as mpatches

plt.rcParams.update({'font.size': 12})

REF_LENGTH = 29903

# Load secondary structure information, TODO: The illustration in the experimental paper is for Vero cells
df_sec_str = pd.read_csv(
    f'/Users/georgangehrn/Desktop/SARS-CoV2-mut-fitness/sec_stru_data/data/sec_structure_Huh7.txt',
    header=None, sep='\s+').drop(columns=[2, 3, 5])
df_sec_str.rename(columns={0: 'nt_site', 1: 'nt_type', 4: 'unpaired'}, inplace=True)

# Assign a color to every mutation type (cf. parameter plots for general linear model)
mut_type_cols = {'AC': 'blue', 'AG': 'orange', 'AT': 'green', 'CA': 'red', 'CG': 'purple', 'CT': 'brown',
                 'GA': 'pink', 'GC': 'gray', 'GT': 'cyan', 'TA': 'magenta', 'TC': 'lime', 'TG': 'teal'}

mut_type_cols_2 = {'noncoding': 'purple', 'synonymous': 'blue', 'nonsynonymous': 'orange'}


def get_conserved_regions(only_known=False):

    TRS_motif = "ACGAAC"  # consensus sequence of transcription regulatory sites
    margin = 0
    ref = SeqIO.read('../human_data/reference.gb', 'genbank')

    motifs = []
    pos = 1
    while pos > 0:
        pos = ref.seq.find(TRS_motif, pos)
        if pos > 1:
            motifs.append(pos)
            pos += 1

    conserved_regions = {}
    for mi, mpos in enumerate(motifs):
        conserved_regions[f'TRS {mi + 1}'] = (mpos - margin, mpos + len(TRS_motif) + margin)

    # attenuator hairpin
    conserved_regions['hairpin'] = (ref.seq.find("ATGCTTCA"), ref.seq.find("CGTTTTT"))
    # attenuator hairpin
    conserved_regions['slippery seq'] = (ref.seq.find("TTTAAACG"), ref.seq.find("TTTAAACG") + 7)
    # 3 stem pseudoknot
    conserved_regions['3-stem-pseudoknot'] = (ref.seq.find("GCGGTGT"), ref.seq.find("TTTTGA", 13474))
    if not only_known:
        # center of E
        conserved_regions['E-center'] = (26330, 26360)
        # end of M
        conserved_regions['M-end'] = (27170, 27200)

    conserved_vector = np.zeros(len(ref.seq), dtype=bool)
    for r in conserved_regions.values():
        conserved_vector[r[0]:r[1]] = True

    return conserved_vector, conserved_regions


def compute_rolling_average(window_len, sites, values):

    # Get size of half the sliding window
    half_wndw = (window_len - 1) / 2

    # Prepare arrays for nt sites and rolling averages that exclude edges of the genome
    new_sites = np.arange(half_wndw + 1, REF_LENGTH - half_wndw + 1)
    roll_avg = np.zeros(int(REF_LENGTH - 2 * half_wndw))
    n_of_datapoints = np.zeros(int(REF_LENGTH - 2 * half_wndw))

    # Compute the rolling average for every nt site
    for i, site in enumerate(new_sites):

        # Get all sites in the dataset that are within the current window boundaries
        lower_bound = site - half_wndw
        upper_bound = site + half_wndw
        mask = (sites >= lower_bound) & (sites <= upper_bound)

        roll_avg[i] = np.mean(values[mask])
        n_of_datapoints[i] = len(values[mask])

    return new_sites, roll_avg, n_of_datapoints


def make_region_list(arr):
    # Take an array with nucleotide sites and group regions
    regions = []
    start = arr[0]
    for i, site in enumerate(arr[:-1]):
        if arr[i+1] != site + 1:
            end = site
            regions.append((start, end))
            start = arr[i+1]
    regions.append((start, arr[-1]))
    return regions


def plot_rolling_avg(new_sites, roll_avg_uncorr, roll_avg_corr, nt_lims=(1, REF_LENGTH)):

    # Plot fitness along genome
    plt.figure(figsize=(16, 4), dpi=300)
    plt.plot(new_sites, roll_avg_uncorr, label='uncorrected', color='black', alpha=0.7)
    plt.plot(new_sites, roll_avg_corr, label='corrected', color='green')
    plt.ylim((-4.2, 1.2))
    plt.ylabel('fitness effect')
    plt.xlabel('nucleotide position')
    plt.legend(fontsize=7)

    # Identify all regions with spikes by checking which sites are far away from the mean
    mean_corr, std_corr = np.nanmean(roll_avg_corr), np.nanstd(roll_avg_corr)
    mean_uncorr, std_uncorr = np.nanmean(roll_avg_uncorr), np.nanstd(roll_avg_uncorr)
    spiking_regions = make_region_list(np.where(roll_avg_corr < mean_corr - 2 * std_corr)[0] + int(WINDOW_LEN / 2) + 1)
    spiking_regions_uncorr = make_region_list(np.where(roll_avg_uncorr < mean_uncorr - 2 * std_uncorr)[0] + int(WINDOW_LEN / 2) + 1)

    # Add gene boundaries to plot
    colors = cm.get_cmap('tab20', len(gene_boundaries))
    for i, (start, end, name) in enumerate(gene_boundaries):
        if start > nt_lims[0]:
            plt.barh(y=-3.5 + (i % 2) * 0.2, width=end - start, left=start, height=0.2, color=colors(i), alpha=0.7)
            plt.text(start, -3.5 + (i % 2) * 0.2, name, ha='left', va='center', fontsize=10)
        elif end > nt_lims[0]:
            plt.barh(y=-3.5 + (i % 2) * 0.2, width=end - nt_lims[0], left=nt_lims[0], height=0.2, color=colors(i), alpha=0.7)
            plt.text(nt_lims[0], -3.5 + (i % 2) * 0.2, name, ha='left', va='center', fontsize=10)

    # Add spiking regions found with corrected fitness effect
    # TODO: find a better solution for defining the boundaries of the region
    plt.axhline(y=mean_corr, color='green', linestyle='-')
    plt.axhline(y=mean_corr - 2 * std_corr, color='green', linewidth=1, linestyle='--')
    plt.axhline(y=mean_corr + 2 * std_corr, color='green', linewidth=1, linestyle='--')
    for i, (start, end) in enumerate(spiking_regions):
        if start > nt_lims[0]:
            plt.barh(y=-2.2, width=WINDOW_LEN - 1 + end - start, left=start - (WINDOW_LEN - 1) / 2, height=0.2, color='red', alpha=1)

    # Add spiking regions found with uncorrected fitness effects
    plt.axhline(y=mean_uncorr, color='black', alpha=0.7)
    plt.axhline(y=mean_uncorr - 2 * std_uncorr, color='black', linestyle='dashed', linewidth=1, alpha=0.7)
    plt.axhline(y=mean_uncorr + 2 * std_uncorr, color='black', linestyle='dashed', linewidth=1, alpha=0.7)
    for i, (start, end) in enumerate(spiking_regions_uncorr):
        if start > nt_lims[0]:
            plt.barh(y=-2.4, width=WINDOW_LEN - 1 + end - start, left=start - (WINDOW_LEN - 1) / 2, height=0.2, color='orange',
                     alpha=0.7)

    # Add known conserved regions
    _, conserved_regions = get_conserved_regions()
    conserved_regions = [(value[0], value[1], key) for key, value in conserved_regions.items()]
    for i, (start, end, name) in enumerate(conserved_regions):
        if start > nt_lims[0]:
            plt.barh(y=-2.6 - (i % 3) * 0.2, width=WINDOW_LEN - 1 + end - start, left=start - (WINDOW_LEN - 1) / 2, height=0.2, color='pink',
                     alpha=1)
            plt.text(start, -2.6 - (i % 3) * 0.2, name, ha='left', va='center', fontsize=10)

    # Add mean and variance of rolling average
    if nt_lims == (1, REF_LENGTH):
        plt.text(0.5, 0.1, f"$\mu$ = {mean_uncorr:.2f}, var: {np.nanvar(roll_avg_uncorr):.2f}", transform=plt.gca().transAxes, color='black', ha='center', fontsize=10)
        plt.text(0.5, 0.05, f"$\mu$ = {mean_corr:.2f}, var: {np.nanvar(roll_avg_corr):.2f}", transform=plt.gca().transAxes, color='green', ha='center', fontsize=10)

    plt.title(f"clade: {CLADE}, avg. window: {WINDOW_LEN} nts, incl. noncoding: {INCL_NONCODING}, incl. tolerant ORFs: {INCL_TOLERANT}, remove ORF9b: {RM_ORF9A}")
    plt.xlim(nt_lims)
    plt.tight_layout()
    plt.show()

    return spiking_regions


def plot_n_of_datapoints(new_sites, n_of_datapoints):
    # Make plot for full genome
    plt.figure(figsize=(16, 4))
    plt.plot(new_sites, n_of_datapoints)
    plt.xlim((new_sites[0], new_sites[-1]))
    colors = cm.get_cmap('tab20', len(gene_boundaries))
    for i, (start, end, name) in enumerate(gene_boundaries):
        plt.barh(y=10 + (i % 2) * 2, width=end - start, left=start, height=2, color=colors(i), alpha=0.7)
        plt.text(start, 10 + (i % 2) * 2, name, ha='left', va='center', fontsize=10)
    plt.xlabel('nucleotide position')
    plt.ylabel('# of mutations used for rolling average')
    plt.title(f"{CLADE}, window: {WINDOW_LEN} nts")
    plt.tight_layout()
    plt.show()

    # Make plot for structural and accessory genes
    plt.figure(figsize=(16, 4))
    plt.plot(new_sites, n_of_datapoints)
    plt.xlim((new_sites[0], new_sites[-1]))
    colors = cm.get_cmap('tab20', len(gene_boundaries))
    for i, (start, end, name) in enumerate(gene_boundaries):
        if start > 25000:
            plt.barh(y=10 + (i % 2) * 2, width=end - start, left=start, height=2, color=colors(i), alpha=0.7)
            plt.text(start, 10 + (i % 2) * 2, name, ha='left', va='center', fontsize=10)
    plt.xlim((25000, REF_LENGTH))
    plt.xlabel('nucleotide position')
    plt.ylabel('# of mutations used for rolling average')
    plt.title(f"{CLADE}, window: {WINDOW_LEN} nts")
    plt.tight_layout()
    plt.show()


def zoom_into_regions(df_all, regions):
    """
    Parameters:
    df_all (DataFrame): mutation counts data across the whole genome
    regions (list of tuples): list of region boundaries
    """

    # Loop over all regions
    # TODO: Some regions are very close together, solve this problem in an automated way
    for k, region_edges in enumerate(regions):

        # Get region edges (add half the window size at both edges)
        region = (region_edges[0] - half_window, region_edges[1] + half_window)

        # Get corresponding section of the reference sequence
        seq = REF_SEQ[int(region[0]-1):int(region[1])]

        # Extract data for all mutations that fall into this region
        df = df_all[(df_all['nt_site'] >= region[0]) & (df_all['nt_site'] <= region[1])]

        # Extract nucleotide positions and fitness effects from dataframe
        sites = df['nt_site'].values
        fitness = {'corrected': np.log(df['actual_count'].values + 0.5) - df['pred_log_count'].values,
                   'uncorrected': np.log(df['actual_count'].values + 0.5) - np.log(df['expected_count'].values + 0.5)}

        # Only show plot if there is more than one datapoint and more than one data point with fitness < -3
        if np.min(fitness['corrected']) > -2.5:
            print(f'discarded region {k + 1} because minimal fitness too high')
            continue

        # Color datapoints according to mutation type
        mut_type = df['nt_mutation'].apply(lambda x: x[0] + x[-1])
        scatter_colors = mut_type.apply(lambda x: mut_type_cols[x])

        # Get maximal fitness effect in this region to scale the y direction of the plot accordingly
        fit_max = max(np.max(fitness['corrected']), np.max(fitness['uncorrected']),
                      FITNESS_AVG['corrected'] + FITNESS_STD['corrected'])

        # Get secondary structure information
        df_help = df_sec_str[(df_sec_str['nt_site'] >= region[0]) & (df_sec_str['nt_site'] <= region[1])]
        sites_help = df_help['nt_site'].values
        unpaired = (df_help['unpaired'].values == 0)
        paired = 1 - unpaired

        # Create a plot for both the corrected and uncorrected fitness effect
        for type in ['uncorrected', 'corrected']:

            # Set up plots with scaled sizes
            # TODO: In principle, the horizontal bars and the text also need to be scaled
            w = 0.125 * (region[1] - region[0])
            h = 0.6 * (fit_max - FITNESS_MIN)
            plt.figure(figsize=(w, h))

            # Add pairing information
            plt.bar(sites_help, unpaired * 0.2, bottom=1.1 * FITNESS_MIN + 0.5, color='red', alpha=0.7)
            plt.bar(sites_help, paired * 0.2, bottom=1.1 * FITNESS_MIN + 0.5, color='blue', alpha=0.7)

            # Add average fitness and standard deviations
            plt.axhline(0, linestyle='-', color='gray', lw=2)
            plt.axhline(FITNESS_AVG[type], linestyle='-', color=COLOR[type], lw=1)
            plt.axhline(FITNESS_AVG[type] + FITNESS_STD[type], linestyle='--', color=COLOR[type], lw=1)
            plt.axhline(FITNESS_AVG[type] - FITNESS_STD[type], linestyle='--', color=COLOR[type], lw=1)

            # Add parent nucleotides
            sites_help = np.arange(region[0], region[1] + 1)
            for i, letter in enumerate(seq):
                plt.text(sites_help[i], 1.1 * FITNESS_MIN + 0.1, letter, ha='center', va='bottom')

            # Add gene boundaries
            gene_colors = cm.get_cmap('tab20', len(gene_boundaries))
            for i, (start, end, name) in enumerate(gene_boundaries):
                if ((start <= region[0] <= end) or (start <= region[1] <= end)) or ((region[0] <= start <= region[1]) and (region[0] <= start <= region[1])):
                    start = max(start, region[0])
                    end = min(end, region[1])
                    plt.barh(y=1.1 * FITNESS_MIN + 0.9 + (i % 2) * 0.3, width=end - start, left=start, height=0.3, color=gene_colors(i), alpha=0.6)
                    plt.text(start, 1.1 * FITNESS_MIN + 0.9 + (i % 2) * 0.3, name, ha='left', va='center', fontsize=10)

            # Add known conserved regions
            _, conserved_regions = get_conserved_regions()
            conserved_regions = [(value[0], value[1], key) for key, value in conserved_regions.items()]
            for i, (start, end, name) in enumerate(conserved_regions):
                if ((start <= region[0] <= end) or (start <= region[1] <= end)) or ((region[0] <= start <= region[1]) and (region[0] <= start <= region[1])):
                    plt.barh(y=1.1 * FITNESS_MIN + 1.5 + (i % 3) * 0.3, width=end - start, left=start, height=0.3, color='orange', alpha=0.6)
                    plt.text(max(start, region[0]), 1.1 * FITNESS_MIN + 1.5 + (i % 3) * 0.3, name, ha='left', va='center', fontsize=10)

            # Set x-limits to region boundaries and y-limits to fitness minimum/maximum across corrected/uncorrected
            plt.xlim((region[0], region[1]))
            plt.ylim((1.1 * FITNESS_MIN, 1.1 * fit_max))

            # Add scatter plot
            plt.scatter(sites, fitness[type], c=scatter_colors, s=60, edgecolors='black', alpha=0.8)

            # Add x-tick labels
            xticks = range(min(sites) + 10 - min(sites) % 10, max(sites) + 1, 10)
            plt.xticks(xticks)

            # Add vertical grid lines
            plt.minorticks_on()
            plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
            plt.grid(which='major', axis='x', linestyle='-', linewidth=0.5, color='black')
            plt.grid(which='minor', axis='x', linestyle=':', linewidth=0.5, color='gray')

            # Add legend for mutation type colors
            legend_handles = [mpatches.Patch(color=color, label=mut_type) for mut_type, color in mut_type_cols.items()]
            plt.legend(handles=legend_handles, fontsize='xx-small', loc='lower right')

            # Add x- and y-labels
            plt.ylabel('fitness effect')
            plt.xlabel('nucleotide site')

            # Add title
            plt.title(f"{type} fitness estimate (region {k+1})")

            # Show plot
            plt.tight_layout()
            plt.show()


def zoom_into_regions_w_nonsyn(df_all, regions):
    """
    Parameters:
    df_all (DataFrame): mutation counts data across the whole genome
    regions (list of tuples): list of region boundaries
    """

    # Loop over all regions
    # TODO: Some regions are very close together, solve this problem in an automated way
    for k, region_edges in enumerate(regions):

        # Get region edges (add half the window size at both edges)
        region = (region_edges[0] - half_window, region_edges[1] + half_window)

        # Get corresponding section of the reference sequence
        seq = REF_SEQ[int(region[0]-1):int(region[1])]

        # Extract data for all mutations that fall into this region
        df = df_all[(df_all['nt_site'] >= region[0]) & (df_all['nt_site'] <= region[1])]

        # Extract nucleotide positions and fitness effects from dataframe
        sites = df['nt_site'].values
        fitness = {'corrected': np.log(df['actual_count'].values + 0.5) - df['pred_log_count'].values,
                   'uncorrected': np.log(df['actual_count'].values + 0.5) - np.log(df['expected_count'].values + 0.5)}

        # Only show plot if there is more than one datapoint and more than one data point with fitness < -3
        if np.min(fitness['corrected']) > -2.5 or len(fitness['corrected']) < 4 * WINDOW_LEN:
            print(f'discarded region {k + 1} because minimal fitness too high')
            continue

        # Color points according to whether the mutation is synonymous, noncoding, coding and not snynonymous,
        # and stop codon mutation
        colors = np.full(len(df), "", dtype='<U20')
        colors[df['noncoding'].values == True] = "noncoding"
        colors[df['synonymous'].values == True] = "synonymous"
        colors[colors == ''] = 'nonsynonymous'
        scatter_colors = np.vectorize(mut_type_cols_2.get)(colors)

        # Color datapoints according to mutation type
        # mut_type = df['nt_mutation'].apply(lambda x: x[0] + x[-1])
        # scatter_colors = mut_type.apply(lambda x: mut_type_cols[x])

        # Get maximal fitness effect in this region to scale the y direction of the plot accordingly
        fit_max = max(np.max(fitness['corrected']), np.max(fitness['uncorrected']),
                      FITNESS_AVG['corrected'] + FITNESS_STD['corrected'])

        # Get secondary structure information
        df_help = df_sec_str[(df_sec_str['nt_site'] >= region[0]) & (df_sec_str['nt_site'] <= region[1])]
        sites_help = df_help['nt_site'].values
        unpaired = (df_help['unpaired'].values == 0)
        paired = 1 - unpaired

        # Create a plot for both the corrected and uncorrected fitness effect
        for type in ['uncorrected', 'corrected']:

            # Set up plots with scaled sizes
            # TODO: In principle, the horizontal bars and the text also need to be scaled
            w = 0.125 * (region[1] - region[0])
            h = 0.6 * (fit_max - FITNESS_MIN)
            plt.figure(figsize=(w, h))

            # Add pairing information
            plt.bar(sites_help, unpaired * 0.2, bottom=1.1 * FITNESS_MIN + 0.5, color='red', alpha=0.7)
            plt.bar(sites_help, paired * 0.2, bottom=1.1 * FITNESS_MIN + 0.5, color='blue', alpha=0.7)

            # Add average fitness and standard deviations
            plt.axhline(0, linestyle='-', color='gray', lw=2)
            plt.axhline(FITNESS_AVG[type], linestyle='-', color=COLOR[type], lw=1)
            plt.axhline(FITNESS_AVG[type] + FITNESS_STD[type], linestyle='--', color=COLOR[type], lw=1)
            plt.axhline(FITNESS_AVG[type] - FITNESS_STD[type], linestyle='--', color=COLOR[type], lw=1)

            # Add parent nucleotides
            sites_help = np.arange(region[0], region[1] + 1)
            for i, letter in enumerate(seq):
                plt.text(sites_help[i], 1.1 * FITNESS_MIN + 0.1, letter, ha='center', va='bottom')

            # Add gene boundaries
            gene_colors = cm.get_cmap('tab20', len(gene_boundaries))
            for i, (start, end, name) in enumerate(gene_boundaries):
                if ((start <= region[0] <= end) or (start <= region[1] <= end)) or ((region[0] <= start <= region[1]) and (region[0] <= start <= region[1])):
                    start = max(start, region[0])
                    end = min(end, region[1])
                    plt.barh(y=1.1 * FITNESS_MIN + 0.9 + (i % 2) * 0.3, width=end - start, left=start, height=0.3, color=gene_colors(i), alpha=0.6)
                    plt.text(start, 1.1 * FITNESS_MIN + 0.9 + (i % 2) * 0.3, name, ha='left', va='center', fontsize=10)

            # Add known conserved regions
            _, conserved_regions = get_conserved_regions()
            conserved_regions = [(value[0], value[1], key) for key, value in conserved_regions.items()]
            for i, (start, end, name) in enumerate(conserved_regions):
                if ((start <= region[0] <= end) or (start <= region[1] <= end)) or ((region[0] <= start <= region[1]) and (region[0] <= start <= region[1])):
                    plt.barh(y=1.1 * FITNESS_MIN + 1.5 + (i % 3) * 0.3, width=end - start, left=start, height=0.3, color='orange', alpha=0.6)
                    plt.text(max(start, region[0]), 1.1 * FITNESS_MIN + 1.5 + (i % 3) * 0.3, name, ha='left', va='center', fontsize=10)

            # Set x-limits to region boundaries and y-limits to fitness minimum/maximum across corrected/uncorrected
            plt.xlim((region[0], region[1]))
            plt.ylim((1.1 * FITNESS_MIN, 1.1 * fit_max))

            # Add scatter plot
            plt.scatter(sites, fitness[type], c=scatter_colors, s=60, edgecolors='black', alpha=0.8)

            # Add x-tick labels
            xticks = range(min(sites) + 10 - min(sites) % 10, max(sites) + 1, 10)
            plt.xticks(xticks)

            # Add vertical grid lines
            plt.minorticks_on()
            plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
            plt.grid(which='major', axis='x', linestyle='-', linewidth=0.5, color='black')
            plt.grid(which='minor', axis='x', linestyle=':', linewidth=0.5, color='gray')

            # Add legend for mutation type colors
            legend_handles = [mpatches.Patch(color=color, label=mut_type) for mut_type, color in mut_type_cols_2.items()]
            plt.legend(handles=legend_handles, fontsize='xx-small', loc='lower right')

            # Add x- and y-labels
            plt.ylabel('fitness effect')
            plt.xlabel('nucleotide site')

            # Add title
            plt.title(f"{type} fitness estimate (region {k+1})")

            # Show plot
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':

    CLADE = '21J'  # Select clade, TODO: Use the same data as Seattle

    WINDOW_LEN = 51  # Define size of averaging window
    half_window = (WINDOW_LEN - 1) / 2

    INCL_NONCODING = True  # Include sites that are labeled as noncoding
    INCL_TOLERANT = True  # Include ORFs that are tolerant to stop codons according to J. Bloom and R. Neher (2023)
    RM_ORF9A = True  # Remove ORF9a to have data for that region
    INCL_ORF1AB_OVERLAP = 'both'  # Include mutations within the overlap of ORF1a and ORF1b which are synonymous in
    # one of the two reading frames

    PLOT_N_DATAPOINTS = False

    NONSYN = True  # if True, all non-excluded mutations are plotted

    # Read in 21J reference sequence, TODO: Find a better solution that not only works for 21J
    with open(f'{CLADE}_refseq', 'r') as file:
        REF_SEQ = file.read()

    # Load mutation counts data
    df = load_mut_counts(clade=CLADE, include_noncoding=INCL_NONCODING, include_tolerant_orfs=INCL_TOLERANT,
                         remove_orf9b=RM_ORF9A, incl_orf1ab_overlap=INCL_ORF1AB_OVERLAP)

    # Add predicted ln(counts + 0.5) to dataframe
    df = add_predictions(df.copy(), clade=CLADE)

    # Extract mutation counts
    data_sites = df['nt_site'].values
    obs_log_counts = np.log(df['actual_count'].values + 0.5)
    exp_log_counts = np.log(df['expected_count'].values + 0.5)
    pred_log_counts = df['pred_log_count'].values

    # Compute uncorrected and corrected fitness estimate
    fitness_uncorr = obs_log_counts - exp_log_counts
    fitness_corr = obs_log_counts - pred_log_counts

    # Characterise distribution of corrected fitness
    FITNESS_AVG = {'corrected': np.average(fitness_corr), 'uncorrected': np.average(fitness_uncorr)}
    FITNESS_STD = {'corrected': np.std(fitness_corr), 'uncorrected': np.std(fitness_uncorr)}
    FITNESS_MAX = {'corrected': np.max(fitness_corr), 'uncorrected': np.max(fitness_uncorr)}
    FITNESS_MIN = min(np.min(fitness_uncorr), np.min(fitness_corr))
    COLOR = {'corrected': 'green', 'uncorrected': 'black'}

    # Compute rolling averages of fitness
    all_sites, roll_avg_uncorr, n_of_datapoints = compute_rolling_average(WINDOW_LEN, data_sites, fitness_uncorr)
    _, roll_avg_corr, _ = compute_rolling_average(WINDOW_LEN, data_sites, fitness_corr)

    # Plot fitness across genome and across structural/accessory proteins and extract spiking regions
    spiking_regions = plot_rolling_avg(all_sites, roll_avg_uncorr, roll_avg_corr)
    _ = plot_rolling_avg(all_sites, roll_avg_uncorr, roll_avg_corr, nt_lims=(25000, REF_LENGTH))

    # Plot number of datapoints used in the sliding windows across the genome
    if PLOT_N_DATAPOINTS:
        plot_n_of_datapoints(all_sites, n_of_datapoints)

    # Connect regions whose ends are close to each other
    contracted_regions = []
    i = 0
    while i < len(spiking_regions):
        if i < len(spiking_regions) - 1 and spiking_regions[i][1] > spiking_regions[i + 1][0] - 10:
            contracted_region = (spiking_regions[i][0], spiking_regions[i + 1][1])
            contracted_regions.append(contracted_region)
            i += 2
        else:
            contracted_regions.append(spiking_regions[i])
            i += 1

    # Zoom into the spiking regions identified during plot_rolling_avg()
    if NONSYN:
        df_nonsyn = load_mut_counts(clade=CLADE, mut_types='non-excluded')
        df_nonsyn = add_predictions(df_nonsyn.copy(), clade=CLADE)
        zoom_into_regions_w_nonsyn(df_nonsyn, contracted_regions)
    else:
        zoom_into_regions(df, contracted_regions)
