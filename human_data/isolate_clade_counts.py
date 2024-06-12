import pandas as pd

clade = '21J'

df = pd.read_csv(f'/Users/georgangehrn/Desktop/SARS2-mut-fitness-corr/human_data/counts/counts_all.csv')

df = df[df['clade'] == clade]

df.to_csv(f'counts/counts_all_{clade}.csv', index=False)
