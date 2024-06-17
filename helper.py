import numpy as np
import pandas as pd

letters = ['A', 'C', 'G', 'T']

mut_types = ['AC', 'AG', 'AT', 'CA', 'CG', 'CT', 'GA', 'GC', 'GT', 'TA', 'TC', 'TG']

plot_map = {'AC': (0, 0), 'CA': (0, 1), 'GA': (0, 2), 'TA': (0, 3),
            'AG': (1, 0), 'CG': (1, 1), 'GC': (1, 2), 'TC': (1, 3),
            'AT': (2, 0), 'CT': (2, 1), 'GT': (2, 2), 'TG': (2, 3)
            }

# obtained from get_gene_boundaries.py
gene_boundaries = [(266, 13480, 'ORF1a'),
                   (13468, 21552, 'ORF1b'),
                   (21563, 25381, 'S'),
                   (25393, 26217, 'ORF3a'),
                   (26245, 26469, 'E'),
                   (26523, 27188, 'M'),
                   (27202, 27384, 'ORF6'),
                   (27394, 27756, 'ORF7a'),
                   (27756, 27884, 'ORF7b'),
                   (27894, 28256, 'ORF8'),
                   (28274, 28283, 'N'),
                   (28284, 28574, 'N;ORF9b'),
                   (28575, 29530, 'N'),
                   (29558, 29671, 'ORF10')]

# Open reading frames that are tolerant to stop codons (see Fig. 3B in J. Bloom and R. Neher (2023))
tolerant_orfs = ['ORF6', 'ORF7a', 'ORF7b', 'ORF8', 'ORF10']


def load_mut_counts(clade, mut_types='synonymous', sec_str_cell_type='Huh7', rm_discrepant_contexts=None,
                    include_noncoding=False, remove_orf9b=False, include_tolerant_orfs=False, incl_orf1ab_overlap=False,
                    verbose=True):

    # Load file with all mutation counts in selected clade
    df = pd.read_csv(f'/Users/georgangehrn/Desktop/SARS-CoV2-mut-fitness/human_data/counts/counts_all_{clade}.csv')

    # Add -2 to +2 context
    nts = df['clade_founder_nt'][0::3].values
    context = ['XXXXX'] * len(nts)
    context[2:-2] = nts[0:-4] + nts[1:-3] + nts[2:-2] + nts[3:-1] + nts[4:]
    df['context'] = np.repeat(context, 3)

    # Load secondary structure for selected cell type (29882 nts, missing some A's at the end compared to Wuhan-Hu-1)
    df_sec_str = pd.read_csv(
        f'/Users/georgangehrn/Desktop/SARS-CoV2-mut-fitness/sec_stru_data/data/sec_structure_{sec_str_cell_type}.txt',
        header=None,
        sep='\s+').drop(columns=[2, 3, 5])
    df_sec_str.rename(columns={0: 'nt_site', 1: 'nt_type', 4: 'unpaired'}, inplace=True)

    # Set labels for paired (0), unpaired (1), and no data (-1)
    # TODO: Add option to remove sites where secondary structure prediction is ambiguous between Huh7 and Vero
    unpaired = (df_sec_str['unpaired'] == 0)
    unpaired = np.hstack((unpaired, np.full(len(nts) - len(df_sec_str), -1)))
    df['unpaired'] = np.repeat(unpaired, 3)

    # Get -2 to +2 context from secondary structure data for comparison to the one obtained from counts_all_{clade}.csv
    nts_sec_str = np.hstack((df_sec_str['nt_type'], np.full(len(nts) - len(df_sec_str), 'N')))
    context_sec_str = ['XXXXX'] * len(nts_sec_str)
    context_sec_str[2:-2] = nts_sec_str[0:-4] + nts_sec_str[1:-3] + nts_sec_str[2:-2] + nts_sec_str[3:-1] + nts_sec_str[4:]
    context_sec_str = np.repeat(context_sec_str, 3)

    # Remove sites with discrepant context information
    if rm_discrepant_contexts is not None:

        [start, end] = rm_discrepant_contexts

        context = df['context'].apply(lambda x: x[2 + start: 2 + end + 1]).values
        context_sec_str = np.array([s[2 + start: 2 + end + 1] for s in context_sec_str])

        df = df[context == context_sec_str]

        if verbose:
            print(f'{int(np.count_nonzero(context != context_sec_str)/3)} sites excluded because contexts not '
                  f'consistent between data sources.\n')

    # Mask for non-excluded mutations/sites
    mask_non_excluded = (df['exclude'] == False).values

    # Mask for mutations that are labeled as synonymous
    mask_synonymous = (df['synonymous'] == True).values

    # Mask for mutations that are labeled as four-fold-degenerate
    mask_ffd = (df['four_fold_degenerate'] == True).values

    # Mask for noncoding sites
    mask_noncoding = (df['noncoding'] == True).values

    # Mask for sites which are in the stop codon tolerant open reading frames
    pattern = '|'.join(tolerant_orfs)
    mask_tolerant = (df['gene'].str.contains(pattern)).values

    # Mask for mutations that are in N;ORF9b and are synonymous in N
    mask_orf9b = ((df['gene'] == 'N;ORF9b') & (df['clade_founder_aa'].apply(lambda x: x[0]) == df['mutant_aa'].apply(lambda x: x[0]))).values

    # Mask for the overlapping region of ORF1a and ORF1b
    if incl_orf1ab_overlap == 'ORF1a':
        mask_orf1ab_overlap = ((df['nt_site'] >= 13469) & (df['nt_site'] <= 13480) & (df['clade_founder_aa'].apply(lambda x: x[0]) == df['mutant_aa'].apply(lambda x: x[0]))).values
    elif incl_orf1ab_overlap == 'ORF1b':
        mask_orf1ab_overlap = ((df['nt_site'] >= 13469) & (df['nt_site'] <= 13480) & (df['clade_founder_aa'].apply(lambda x: x[-1]) == df['mutant_aa'].apply(lambda x: x[-1]))).values
    elif incl_orf1ab_overlap == 'both':
        mask_orf1ab_overlap = ((df['nt_site'] >= 13469) & (df['nt_site'] <= 13480) & (df['clade_founder_aa'].apply(lambda x: x[-1]) == df['mutant_aa'].apply(lambda x: x[-1])) | (df['clade_founder_aa'].apply(lambda x: x[0]) == df['mutant_aa'].apply(lambda x: x[0]))).values

    # Choose between synonymous and four-fold-degenerate mutations
    mask_inter = mask_synonymous
    if mut_types == 'four_fold_degenerate':
        mask_inter = mask_ffd

    # Add other mutation types
    if include_noncoding:
        mask_inter = mask_inter + mask_noncoding
    if include_tolerant_orfs:
        mask_inter = mask_inter + mask_tolerant
    if remove_orf9b:
        mask_inter = mask_inter + mask_orf9b
    if incl_orf1ab_overlap:
        mask_inter = mask_inter + mask_orf1ab_overlap

    if mut_types == 'non-excluded':
        return df[mask_non_excluded]
    else:
        return df[mask_non_excluded * mask_inter]
