import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

if __name__ == '__main__':

    ''' This script takes the raw data from Bloom et al. (2024) 'counts/counts.csv',
    adds secondary structure + DMS value + context information, 
    and saves a file 'counts/counts_all_{clade}_nonex_{syn}.csv'.
    
    parameters: clade, syn, cell_type '''

    # Define clade to be extracted
    clade = '21J'

    # Define which type of mutation to extract
    syn = 'syn'

    # Define which secondary structure information should be included
    cell_type = ['Huh7', 'Vero'][0]

    # Define whether to visualise excluded sites
    show_excl = False

    # Load all mutation data and only keep one clade
    df = pd.read_csv('../counts/counts_all.csv')
    df = df[(df['clade'] == clade)]

    # Visualise number of excluded mutations per site across genome
    num_excl_muts = np.sum(df['exclude'].values.reshape(29903, 3), axis=1)
    site_num = np.array_split(np.arange(len(num_excl_muts)), 3)
    num_excl_muts = np.array_split(num_excl_muts, 3)
    fig, axes = plt.subplots(3, 1, figsize=(20, 10))
    for i, ax in enumerate(axes.flat):
        ax.scatter(site_num[i], num_excl_muts[i], s=15)
        if i == 2:
            ax.set_xlabel('nucleotide position')
        ax.set_ylabel('no. of excl. muts.')
        ax.text(np.median(site_num[i]), 2, 'total for this region: ' + str(np.sum(num_excl_muts[i])) + '/'
                + str((len(site_num[i]) * 3)))
    fig.suptitle(clade, fontsize=16)
    plt.tight_layout()
    plt.savefig('excluded_mutations.png')
    plt.show() if show_excl else plt.close()

    '''
    Frameshift:
    As far as I understand it, the ribosome runs either until nt 13'480 and stops (ORF1a (where is
    the stop codon?)) or reads 13'468 twice (once as 3rd nt in codon and once as 1st) and then runs until nt 21552
    and stops (ORF1ab). So there is a region (13468 until 13480) where it is actually ambiguous whether a given nt
    mutation is synonymous or not (I still include the given synonymous information).
    '''

    # Load secondary structure data (the 21J reference sequence has some extra nucleotides at the end which is why it
    # is a bit longer than the sequence from the Huh7 dataset)
    df_stru = pd.read_csv('/Users/georgangehrn/Desktop/SARS2-mut-fitness-corr/sec_stru_data/data/sec_structure_' + cell_type + '.txt',
                          header=None, sep='\s+').drop(columns=[2, 3, 5])
    df_stru.rename(columns={0: 'nt_site', 1: 'nt_type', 4: 'unpaired'}, inplace=True)

    df_stru['unpaired'].values[df_stru['unpaired'].values != 0] = 1
    df_stru['unpaired'] = 1 - df_stru['unpaired'].values

    # Add structural information (pad tail of clade founder with structural information 2)
    clade_founder = df['clade_founder_nt'].values[0::3]
    ref_seq = df_stru['nt_type'].values
    pad_length = len(clade_founder) - len(ref_seq)
    num_diffs = np.count_nonzero(clade_founder[:-pad_length] != ref_seq)
    print(f'{num_diffs}/{len(clade_founder)} nucleotides do not agree between the ' + clade + ' clade founder and the '
          + cell_type + ' reference sequence. These sites are not excluded from the data.')
    print('The ' + clade + f' clade founder is {pad_length} nucleotides longer than the ' + cell_type +
          ' reference sequence. "unpaired" is set equal to 2 for these nucleotides.')
    unpaired = np.hstack((df_stru['unpaired'].values, np.full(pad_length, 2)))
    df['unpaired'] = np.repeat(unpaired, 3)

    # Drop unnecessary columns to avoid confusion
    df.drop(columns=['clade', 'subset', 'masked_in_usher', 'gene', 'clade_founder_codon', 'clade_founder_aa',
                     'mutant_codon', 'mutant_aa', 'aa_mutation', 'noncoding', 'codon_position', 'codon_site',
                     'four_fold_degenerate', 'mean_log_size'], inplace=True)

    # Add context (-2 to +2) information to dataframe
    nts = df.drop_duplicates(subset=['nt_site'])['clade_founder_nt'].values
    context = ['XXXXX'] * len(nts)
    context[2:-2] = nts[0:-4] + nts[1:-3] + nts[2:-2] + nts[3:-1] + nts[4:]
    context = np.repeat(context, 3)
    df['context'] = context

    # Load data and keep only sites which are nt_x
    #dms = pd.read_excel('/Users/georgangehrn/Desktop/SARS2-mut-fitness-corr/sec_stru_data/data/dms_data.xlsx')

    # Remove rows without DMS count
    #df['dms'] = dms.values

    # Only keep non-/synonymous and non-excluded sites
    df = df[(df['exclude'] == False) &
            (df['synonymous'] == {'syn': True, 'nonsyn': False, '': (True or False)}[syn])]
    df.drop(columns=['exclude', 'synonymous'], inplace=True)

    # Save as new csv file
    df.to_csv('../counts/counts_all_' + clade + '_nonex_' + syn + '.csv', index=False, header=True)
