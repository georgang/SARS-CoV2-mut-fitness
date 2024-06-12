import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

plt.rcParams.update({'font.size': 14})

# TODO: Improve the structure of this code and add genes to the figure

# Read in Huh7 structural data
df_huh7 = pd.read_csv('data/sec_structure_Huh7.txt', header=None, sep='\s+').drop(columns=[2, 3, 5])
df_huh7.rename(columns={0: 'nt_site', 1: 'nt_type', 4: 'unpaired'}, inplace=True)

df_huh7['unpaired'].values[df_huh7['unpaired'].values != 0] = 1
df_huh7['unpaired'] = 1 - df_huh7['unpaired'].values

# unpaired_mask = df_huh7['unpaired'].values
# np.savetxt("/human_data/unpaired_mask.csv", unpaired_mask, delimiter=",",
#            fmt='%d')

# Read in Vero structural data
df_vero = pd.read_csv('data/sec_structure_Vero.txt', header=None, sep='\s+').drop(columns=[2, 3, 5])
df_vero.rename(columns={0: 'nt_site', 1: 'nt_type', 4: 'unpaired'}, inplace=True)

df_vero['unpaired'].values[df_vero['unpaired'].values != 0] = 1
df_vero['unpaired'] = 1 - df_vero['unpaired'].values

# Plot differences
huh7 = df_huh7['unpaired'].values
vero = df_vero['unpaired'].values
test_1 = np.array_split((huh7 - vero == 1), 3)
test_2 = np.array_split((huh7 - vero == -1), 3)
site_num = np.array_split(np.arange(len(huh7)), 3)
fig, axes = plt.subplots(3, 1, figsize=(20, 10))
for i, ax in enumerate(axes.flat):
    ax.bar(site_num[i], test_1[i], color='red', width=1, label='unpaired in Huh7, paired in Vero')
    ax.bar(site_num[i], test_2[i], color='blue', width=1, label='paired in Huh7, unpaired in Vero')
    if i == 2:
        ax.set_xlabel('nucleotide position')
    ax.set_ylim((0, 1))
    ax.set_yticks([])
    if i == 0:
        ax.legend()
plt.tight_layout()
plt.savefig('results/differences_huh7_vero.png')
plt.show()

# Calculate confusion matrix
conf_mat = confusion_matrix(huh7, vero)
print('paired in both Huh7 and Vero: ' + str(conf_mat[0, 0]))
print('unpaired in both Huh7 and Vero: ' + str(conf_mat[1, 1]))
print('paired in Huh7, but unpaired in Vero: ' + str(conf_mat[0, 1]))
print('paired in Vero, but unpaired in Huh7: ' + str(conf_mat[1, 0]))

# Compare DMS data
df_dms = pd.read_excel('/Users/georgangehrn/Desktop/SARS2-mut-fitness-corr/sec_stru_data/dms_data.xlsx')

# Get data for the two cell types
dms_huh7 = df_dms['Huh7 (filtered)'].values
dms_vero = df_dms['Vero (filtered)'].values

# Only keep sites which are in both dataframes
mask = ~np.isnan(dms_huh7) & ~np.isnan(dms_vero)

# Filter the arrays using the mask
dms_huh7 = dms_huh7[mask]
dms_vero = dms_vero[mask]

# Scatter against each other
plt.scatter(dms_huh7, dms_vero, s=2)
plt.xlabel('DMS value in Huh7')
plt.ylabel('DMS value in Vero')
plt.text(0.8 * np.max(dms_huh7), 0.8 * np.max(dms_vero),
         'r = ' + str(np.round(np.corrcoef(dms_huh7, dms_vero)[0, 1], 3)))
plt.tight_layout()
plt.grid()
plt.savefig('results/dms_corr_huh7_vero.png')
plt.show()
