import os
import pandas as pd
import numpy as np

# Prepare a dataframe that contains the clade founder nucleotides for all 36 clades
fitness_results_dir = 'results_gisaid_2024-04-24'
founder_df = pd.read_csv(os.path.join(fitness_results_dir, 'clade_founder_nts/clade_founder_nts.csv'))
founder_df.sort_values(['clade', 'site'], inplace=True)

# Get the clade founder sequences for all 36 clades
founder_seq_dict = {}
for (clade, data) in founder_df.groupby('clade'):
    founder_seq_dict[clade] = ''.join(data['nt'])


def get_motif(site, clade):
    founder_seq = founder_seq_dict[clade]
    return founder_seq[site - 2:site + 1]


# Get the -1 to +1 context for every site in every clade
min_and_max_sites = [founder_df['site'].min(), founder_df['site'].max()]
founder_df['motif'] = founder_df.apply(lambda row: np.nan if row['site'] in min_and_max_sites else get_motif(row['site'], row['clade']), axis=1)

# Add columns giving the reference codon and motif (19A serves as the reference!)
founder_df = founder_df.merge((founder_df[founder_df['clade'] == '19A'].rename(columns={'codon': 'ref_codon', 'motif': 'ref_motif'}))[['site', 'ref_codon', 'ref_motif']], on='site', how='left')

# Identify sites where the codon and motif are conserved across all clade founders by subsetting data to entries with
# identical codons/motifs to reference, then identifying sites that still have entries for all clades
data = founder_df[
    (founder_df['codon'] == founder_df['ref_codon']) &
    (founder_df['motif'] == founder_df['ref_motif'])
]
site_counts = data['site'].value_counts()
nclades = len(founder_df['clade'].unique())
conserved_sites = site_counts[site_counts == nclades].index
founder_df['same_context_all_founders'] = founder_df['site'].isin(conserved_sites)
founder_df['nt_site'] = founder_df['site']

print('Number of sites in genome:', len(founder_df['site'].unique()))
print('Number of sites with the same -1 to +1 context in all clades:', len(conserved_sites))

# Read in all data
counts_df = pd.read_csv(os.path.join(
    fitness_results_dir,
    'expected_vs_actual_mut_counts/expected_vs_actual_mut_counts.csv'
))

# Add mutation type X -> Y
counts_df[['wt_nt', 'mut_nt']] = counts_df['nt_mutation'].str.extract(r'(\w)\d+(\w)')
counts_df['mut_type'] = counts_df['wt_nt'] + counts_df['mut_nt']


def get_mut_class(row):
    if row['synonymous']:
        return 'synonymous'
    elif row['noncoding']:
        return 'noncoding'
    elif '*' in row['mutant_aa']:
        return 'nonsense'
    elif row['mutant_aa'] != row['clade_founder_aa']:
        return 'nonsynonymous'
    else:
        raise ValueError(row['mutant_aa'], row['clade_founder_aa'])


# Add mutation class
counts_df['mut_class'] = counts_df.apply(lambda row: get_mut_class(row), axis=1)

# Add column indicating if clade is pre-Omicron or Omicron
pre_omicron_clades = [
    '20A', '20B', '20C', '20E', '20G', '20H', '20I', '20J', '21C','21I', '21J'
]
counts_df['pre_omicron_or_omicron'] = counts_df['clade'].apply(
    lambda x: 'pre_omicron' if x in pre_omicron_clades else 'omicron'
)

# Add column indicating if a site is before site 21,555
counts_df['nt_site_before_21555'] = counts_df['nt_site'] < 21555

# Add column indicating whether RNA sites from the Lan, 2022, Nature Comm. structure are predicted to be paired,
# using code from Hensel, 2023, biorxiv
filename = '41467_2022_28603_MOESM11_ESM.txt'
with open(filename) as f:
    lines = [line.rstrip().split() for line in f]
paired = np.array([[int(x[0]), int(x[4])] for x in lines[1:]])
paired_dict = dict(zip(paired[:, 0], paired[:, 1]))


def assign_ss_pred(site):
    if site not in paired_dict:
        return 'nd'
    elif paired_dict[site] == 0:
        return 'unpaired'
    else:
        return 'paired'


counts_df['ss_prediction'] = counts_df['nt_site'].apply(lambda x: assign_ss_pred(x))

# Add columns giving a site's motif relative to the clade founder and the reference sequence
counts_df = counts_df.merge(
    founder_df[['nt_site', 'clade', 'motif', 'ref_motif']],
    on=['nt_site', 'clade'], how='left',
)

# Ignore sites that are masked or excluded in any clade of the UShER tree
sites_to_ignore = list(counts_df[
    (counts_df['masked_in_usher'] == True) |
    (counts_df['exclude'] == True)
]['nt_site'].unique())

# Homoplastic sites from De Maio et al., which we will also ignore
sites_to_ignore += [
    187, 1059, 2094, 3037, 3130, 6990, 8022, 10323, 10741, 11074, 13408,
    14786, 19684, 20148, 21137, 24034, 24378, 25563, 26144, 26461, 26681, 28077,
    28826, 28854, 29700, 4050, 13402, 11083, 15324, 21575
]

# Aggregate counts across ... TODO: Have a closer look at this!
# ... all clades for "all" subset
ignore_cols = [
    'expected_count', 'actual_count', 'count_terminal', 'count_non_terminal', 'mean_log_size',
    'clade', 'pre_omicron_or_omicron'
]
groupby_cols = [
    col for col in counts_df.columns.values
    if col not in ignore_cols
]
curated_counts_df = counts_df[
    (counts_df['nt_site'].isin(conserved_sites)) &
    ~(counts_df['nt_site'].isin(sites_to_ignore)) &
    (counts_df['subset'] == 'all')
].groupby(groupby_cols, as_index=False).agg('sum', numeric_only=True)
assert sum(curated_counts_df['nt_mutation'].duplicated(keep=False)) == 0

# ... England or USA, and merge counts column with above dataframe
subsets = ['England', 'USA']
for subset in subsets:
    subset_data = counts_df[
        (counts_df['nt_site'].isin(conserved_sites)) &
        ~(counts_df['nt_site'].isin(sites_to_ignore)) &
        (counts_df['subset'] == subset)
    ].groupby(groupby_cols, as_index=False).agg('sum', numeric_only=True)
    assert sum(subset_data['nt_mutation'].duplicated(keep=False)) == 0
    assert len(subset_data) == len(curated_counts_df)
    curated_counts_df = curated_counts_df.merge(
        (
            subset_data
            .rename(columns={'actual_count' : f'actual_count_{subset}'})
        )[['nt_mutation', f'actual_count_{subset}']], on='nt_mutation'
    )

# ... pre-Omicron or Omicron clades, and merge counts column with above dataframe
subsets = ['pre_omicron', 'omicron']
for subset in subsets:
    subset_data = counts_df[
        (counts_df['nt_site'].isin(conserved_sites)) &
        ~(counts_df['nt_site'].isin(sites_to_ignore)) &
        (counts_df['subset'] == 'all') &
        (counts_df['pre_omicron_or_omicron'] == subset)
    ].groupby(groupby_cols, as_index=False).agg('sum', numeric_only=True)
    assert sum(subset_data['nt_mutation'].duplicated(keep=False)) == 0
    assert len(subset_data) == len(curated_counts_df)
    curated_counts_df = curated_counts_df.merge(
        (
            subset_data
            .rename(columns={'actual_count' : f'actual_count_{subset}'})
        )[['nt_mutation', f'actual_count_{subset}']], on='nt_mutation'
    )

# Save curated counts to an output file
assert sum(curated_counts_df['motif'] != curated_counts_df['ref_motif']) == 0
assert len(curated_counts_df) == len(curated_counts_df['nt_mutation'].unique())
curated_counts_df.drop(columns=['subset', 'exclude', 'masked_in_usher'], inplace=True)
outfile = 'curated_mut_counts.csv'
curated_counts_df.to_csv(outfile, index=False)

# Print head of dataframe for quick check
print(curated_counts_df.head())

print('Number of unique muts:')
print('In the full dataset:', len(counts_df['nt_mutation'].unique()))
print('In the curated dataset:', len(curated_counts_df['nt_mutation'].unique()))

print('Number of curated mutations per category:')
curated_counts_df['mut_class'].value_counts()

# Get gene boundaires
gene_boundaries_df = counts_df.groupby('gene', as_index=False).agg(
    min_site=('nt_site', 'min'),
    max_site=('nt_site', 'max'),
)
gene_boundaries_df['gene'].replace('ORF1a;ORF1ab', 'ORF1a', inplace=True)
gene_boundaries_df['gene'].replace('ORF1ab', 'ORF1b', inplace=True)
gene_boundaries_df = gene_boundaries_df[
    ~(gene_boundaries_df['gene'].str.contains(';')) &
    ~(gene_boundaries_df['gene'].isin(['noncoding']))
].reset_index(drop=True).sort_values('min_site')

# Save list to file
outfile = 'gene_boundaries.csv'
gene_boundaries_df.to_csv(outfile, index=False)

print(gene_boundaries_df)
