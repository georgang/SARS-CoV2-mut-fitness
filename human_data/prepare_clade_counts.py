import pandas as pd

'''This script takes counts_all.csv, extracts the data of CLADE, and stores a file "counts_all_{CLADE}.csv".'''

CLADE = '21J'

# Load counts_all.csv
df = pd.read_csv(f'/Users/georgangehrn/Desktop/SARS-CoV2-mut-fitness/human_data/counts/counts_all.csv')

# Isolate data of one clade
df = df[df['clade'] == CLADE]

# Save dataframe
df.to_csv(f'counts/counts_all_{CLADE}.csv', index=False)
