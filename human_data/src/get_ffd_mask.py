import numpy as np
import pandas as pd

if __name__ == '__main__':
    # Define clade to be extracted
    clade = '21J'

    # Load all mutation data and only keep one clade
    df = pd.read_csv('../counts/counts_all.csv')
    df = df[(df['clade'] == clade)]

    # Create mask for four-fold-degenerate sites
    ffd_mask = df['four_fold_degenerate'].values[0::3].astype(int).reshape(-1, 1)

    # Save as new csv file
    np.savetxt("/human_data/ffd_mask.csv", ffd_mask, delimiter=",", fmt='%d')
