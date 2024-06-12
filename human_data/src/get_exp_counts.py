import numpy as np
import pandas as pd

'''
    This script takes the mutation counts and writes the mean values into 'exp_[log]_counts_{clade}.csv'
'''

# Define clade
clade = '21K'

# Load data
df_mut = pd.read_csv('counts/counts_all_' + clade + '_nonex_syn.csv')

# Choose between log and absolute counts
log = True
if log:
    df_mut['actual_count'] = df_mut['actual_count'].apply(lambda x: np.log(x + 0.5))
    df_mut['expected_count'] = df_mut['expected_count'].apply(lambda x: np.log(x + 0.5))

# Define dict to store expected counts
d = {}

# Loop through all mutation types and extract data
letters = ['A', 'C', 'G', 'T']
for nt1 in letters:
    for nt2 in letters:
        if nt1 != nt2:
            # Only keep one mutation type
            df_mut_h = df_mut[df_mut['nt_mutation'].str.match('^' + nt1 + '.*' + nt2 + '$')]

            # Calculate various expected count information for all sites
            exp_count = np.mean(df_mut_h['expected_count'])
            m_syn = np.mean(df_mut_h['actual_count'])
            md_syn = np.median(df_mut_h['actual_count'])

            # Calculate expected count information for paired (p) sites
            m_syn_p = np.mean(df_mut_h[df_mut_h['unpaired'] == 0]['actual_count'])
            md_syn_p = np.median(df_mut_h[df_mut_h['unpaired'] == 0]['actual_count'])

            # Calculate expected count information for unpaired (up) sites
            m_syn_up = np.mean(df_mut_h[df_mut_h['unpaired'] == 1]['actual_count'])
            md_syn_up = np.median(df_mut_h[df_mut_h['unpaired'] == 1]['actual_count'])

            d[nt1 + nt2] = [exp_count,
                            m_syn, md_syn,
                            m_syn_p, md_syn_p,
                            m_syn_up, md_syn_up
                            ]

# Store data
d['type'] = ['exp_count',
             'm_syn', 'md_syn',
             'm_syn_p', 'md_syn_p',
             'm_syn_up', 'md_syn_up'
             ]
df = pd.DataFrame(d)
df.set_index('type', inplace=True)
df.to_csv('exp_counts/exp_log_counts_' + clade + '.csv') if log else df.to_csv('exp_counts/exp_counts_' + clade + '.csv')
