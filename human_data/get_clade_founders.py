import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

if __name__ == '__main__':
    # Load all mutation data and only keep one clade
    df = pd.read_csv('counts/counts_all.csv')
    df_clade = df[(df['clade'] == '19A')]

    print('')
