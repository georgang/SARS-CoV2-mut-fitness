import numpy as np
import pandas as pd

# TODO: Improve this and save the mask somewhere

df_huh7 = pd.read_csv('data/sec_structure_Huh7.txt', header=None, sep='\s+').drop(columns=[2, 3, 5])
df_huh7.rename(columns={0: 'nt_site', 1: 'nt_type', 4: 'unpaired'}, inplace=True)

df_huh7['unpaired'].values[df_huh7['unpaired'].values != 0] = 1
df_huh7['unpaired'] = 1 - df_huh7['unpaired'].values

unpaired_mask = df_huh7['unpaired'].values

unpaired_mask = np.hstack((unpaired_mask, np.full(29903-len(unpaired_mask), 1)))

np.savetxt("/human_data/unpaired_mask.csv", unpaired_mask, delimiter=",",
           fmt='%d')