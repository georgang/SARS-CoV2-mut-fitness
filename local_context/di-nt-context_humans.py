import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})


def load_data(nt_x, nt_y, cell_type, pseudo_count):

    # Read file with mutation data
    df_mut = pd.read_csv('human_data/mut_counts_21J_syn_nonexc.csv')

    # Only keep specified mutation type
    df_mut = df_mut[df_mut['nt_mutation'].str.match('^' + nt_x + '.*' + nt_y + '$')]

    # Load secondary structure and rename columns
    df = pd.read_csv('sec_stru_data/sec_structure_' + cell_type + '.txt', header=None,
                     sep='\s+').drop(columns=[2, 3, 5])
    df.rename(columns={0: 'nt_site', 1: 'nt_type', 4: 'unpaired'}, inplace=True)

    # Add context information
    nts = df['nt_type'].values
    back_context = [''] * len(nts)
    forw_context = [''] * len(nts)
    back_context[0] = 'X'
    back_context[len(nts) - 1] = 'Y'
    forw_context[0] = 'X'
    forw_context[len(nts) - 1] = 'Y'
    back_context[1:-1] = nts[:-2]
    forw_context[1:-1] = nts[2:]
    df['back_context'] = back_context
    df['forw_context'] = forw_context

    # Keep only nt_x sites, and set labels for paired (0) and unpaired (1)
    df = df[df['nt_type'] == nt_x]
    df['unpaired'].values[df['unpaired'].values != 0] = 1
    df['unpaired'] = 1 - df['unpaired'].values

    df = df[df['nt_site'].isin(df_mut['nt_site'])]
    df_mut = df_mut[df_mut['nt_site'].isin(df['nt_site'])]

    df.reset_index(drop=True, inplace=True)
    df_mut.reset_index(drop=True, inplace=True)

    df = df[(df['forw_context'] == df_mut['forw_context'])*(df['back_context'] == df_mut['back_context'])]
    df_mut = df_mut[df_mut['nt_site'].isin(df['nt_site'])]

    # Add actual mutation counts (+ pseudocount) and return expected count
    df.insert(3, 'actual_count', df_mut['actual_count'].values + pseudo_count)
    exp_count = np.mean(df_mut['expected_count'])

    # Load and add DMS data if nt_x is 'C' or 'A'
    if False: #nt_x in ['C', 'A']:

        # Load data and keep only sites which are nt_x
        df_dms = pd.read_excel('sec_stru_data/dms_data.xlsx')
        df_dms = df_dms[df_dms['Nucleotide'] == nt_x]

        # Only keep sites which are also in df
        df_dms = df_dms[df_dms['Genome coordinate'].isin(df['nt_site'])]
        df = df[df['nt_site'].isin(df_dms['Genome coordinate'])]
        df.insert(4, 'dms', df_dms[cell_type + ' (filtered)'])

        # Remove rows without DMS count
        df = df[df['dms'].notnull()]

    return df, exp_count


def context_plot(df, nt_x, nt_y, exp_count, pseudo_count, cont_dir):

    # Extract counts from dataframe
    counts = df['actual_count'].values - pseudo_count

    # Extract counts for paired/unpaired sites dependent on context
    counts_g = counts[(df['unpaired'].values == 0) * (df[cont_dir+'_context'].values == 'G')]
    counts_c = counts[(df['unpaired'].values == 0) * (df[cont_dir+'_context'].values == 'C')]
    counts_a = counts[(df['unpaired'].values == 0) * (df[cont_dir+'_context'].values == 'A')]
    counts_t = counts[(df['unpaired'].values == 0) * (df[cont_dir+'_context'].values == 'T')]

    [min, max] = [np.min(counts), np.max(counts)]
    [lb, ub] = [np.floor(np.percentile(counts, 0.25)), np.ceil(np.percentile(counts, 99))]

    n_bins = int(ub - lb)
    b = round(10 * (ub - lb) / len(counts))
    if (ub - lb) % b != 0 and b != 0:
        ub += b - (ub - lb) % b
        n_bins = int((ub - lb) / b)

    hist, bin_edges = np.histogram(counts, bins=n_bins, range=(lb, ub))
    hist_g, bin_edges_g = np.histogram(counts_g, bins=n_bins, range=(lb, ub))
    hist_c, bin_edges_c = np.histogram(counts_c, bins=n_bins, range=(lb, ub))
    hist_a, bin_edges_a = np.histogram(counts_a, bins=n_bins, range=(lb, ub))
    hist_t, bin_edges_t = np.histogram(counts_t, bins=n_bins, range=(lb, ub))

    hist_max = np.max([np.max(hist_g), np.max(hist_c), np.max(hist_a), np.max(hist_t)])

    fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    # plt.bar(bin_edges[:-1], hist, width=(bin_edges[-1]-bin_edges[0])/n_bins, alpha=0.7, align='edge',
    #         color='none', edgecolor='black')
    axs[0].bar(bin_edges_g[:-1], hist_g, width=(bin_edges_g[-1]-bin_edges_g[0])/n_bins,
             alpha=0.9, align='edge', color='green', label='G')
    axs[1].bar(bin_edges_c[:-1], hist_c, width=(bin_edges[-1] - bin_edges[0]) / n_bins,
            alpha=0.9, align='edge', color='orange', label='C')
    axs[2].bar(bin_edges_a[:-1], hist_a, width=(bin_edges[-1] - bin_edges[0]) / n_bins,
            alpha=0.9, align='edge', color='black', label='A')
    axs[3].bar(bin_edges_t[:-1], hist_t, width=(bin_edges[-1] - bin_edges[0]) / n_bins,
             alpha=0.9, align='edge', color='purple', label='T')
    for ax in axs:
        ax.legend()
        ax.set_ylim(0, hist_max)
    nt_list = [counts_g, counts_c, counts_a, counts_t]
    color_list = ['green', 'orange', 'black', 'purple']
    for i, ax in enumerate(axs, start=1):
        ax.text(0.75, 0.5, 'mean = '+str(round(np.mean(nt_list[i-1]), 3)), transform=ax.transAxes, color=color_list[i-1])
        ax.text(0.75, 0.35, 'median = ' + str(round(np.median(nt_list[i-1]), 3)), transform=ax.transAxes, color=color_list[i-1])
        ax.text(0.75, 0.25, 'N = ' + str(len(nt_list[i - 1])), transform=ax.transAxes, color=color_list[i - 1])
    plt.xlim((lb, ub))
    axs[0].set_title(nt_x + ' --> ' + nt_y + ' (paired)')
    plt.savefig(nt_x + nt_y + '_cont_paired_'+cont_dir+'.png')
    plt.xlabel('mutation counts')
    plt.show()

    # Extract counts from dataframe
    counts = df['actual_count'].values - pseudo_count

    # Extract counts for paired/unpaired sites dependent on context
    counts_g = counts[(df['unpaired'].values == 1) * (df[cont_dir + '_context'].values == 'G')]
    counts_c = counts[(df['unpaired'].values == 1) * (df[cont_dir + '_context'].values == 'C')]
    counts_a = counts[(df['unpaired'].values == 1) * (df[cont_dir + '_context'].values == 'A')]
    counts_t = counts[(df['unpaired'].values == 1) * (df[cont_dir + '_context'].values == 'T')]

    [min, max] = [np.min(counts), np.max(counts)]
    [lb, ub] = [np.floor(np.percentile(counts, 0.25)), np.ceil(np.percentile(counts, 99))]

    n_bins = int(ub - lb)
    b = round(10 * (ub - lb) / len(counts))
    if (ub - lb) % b != 0 and b != 0:
        ub += b - (ub - lb) % b
        n_bins = int((ub - lb) / b)

    hist, bin_edges = np.histogram(counts, bins=n_bins, range=(lb, ub))
    hist_g, bin_edges_g = np.histogram(counts_g, bins=n_bins, range=(lb, ub))
    hist_c, bin_edges_c = np.histogram(counts_c, bins=n_bins, range=(lb, ub))
    hist_a, bin_edges_a = np.histogram(counts_a, bins=n_bins, range=(lb, ub))
    hist_t, bin_edges_t = np.histogram(counts_t, bins=n_bins, range=(lb, ub))

    hist_max = np.max([np.max(hist_g), np.max(hist_c), np.max(hist_a), np.max(hist_t)])

    fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    # plt.bar(bin_edges[:-1], hist, width=(bin_edges[-1]-bin_edges[0])/n_bins, alpha=0.7, align='edge',
    #         color='none', edgecolor='black')
    axs[0].bar(bin_edges_g[:-1], hist_g, width=(bin_edges_g[-1] - bin_edges_g[0]) / n_bins,
               alpha=0.9, align='edge', color='green', label='G')
    axs[1].bar(bin_edges_c[:-1], hist_c, width=(bin_edges[-1] - bin_edges[0]) / n_bins,
               alpha=0.9, align='edge', color='orange', label='C')
    axs[2].bar(bin_edges_a[:-1], hist_a, width=(bin_edges[-1] - bin_edges[0]) / n_bins,
               alpha=0.9, align='edge', color='black', label='A')
    axs[3].bar(bin_edges_t[:-1], hist_t, width=(bin_edges[-1] - bin_edges[0]) / n_bins,
               alpha=0.9, align='edge', color='purple', label='T')
    for ax in axs:
        ax.legend()
        ax.set_ylim(0, hist_max)
    for ax in axs:
        ax.legend()
        ax.set_ylim(0, hist_max)
    nt_list = [counts_g, counts_c, counts_a, counts_t]
    color_list = ['green', 'orange', 'black', 'purple']
    for i, ax in enumerate(axs, start=1):
        ax.text(0.75, 0.5, 'mean = '+str(round(np.mean(nt_list[i-1]), 3)), transform=ax.transAxes, color=color_list[i-1])
        ax.text(0.75, 0.35, 'median = ' + str(round(np.median(nt_list[i-1]), 3)), transform=ax.transAxes, color=color_list[i-1])
        ax.text(0.75, 0.2, 'N = ' + str(len(nt_list[i-1])), transform=ax.transAxes, color=color_list[i-1])
    plt.xlim((lb, ub))
    axs[0].set_title(nt_x + ' --> ' + nt_y + ' (unpaired)')
    plt.savefig(nt_x + nt_y + '_cont_unpaired_'+cont_dir+'.png')
    plt.xlabel('mutation counts')
    plt.show()

    return 0


def context_plot_log(df, nt_x, nt_y, exp_count, pseudo_count, cont_dir):

    # Extract counts from dataframe
    counts = np.log(df['actual_count'].values)

    # Extract counts for paired/unpaired sites dependent on context
    counts_g = counts[(df['unpaired'].values == 0) * (df[cont_dir+'_context'].values == 'G')]
    counts_c = counts[(df['unpaired'].values == 0) * (df[cont_dir+'_context'].values == 'C')]
    counts_a = counts[(df['unpaired'].values == 0) * (df[cont_dir+'_context'].values == 'A')]
    counts_t = counts[(df['unpaired'].values == 0) * (df[cont_dir+'_context'].values == 'T')]

    [log_min, log_max] = [np.min(counts), np.max(counts)]

    [lb, ub] = [np.percentile(counts, 0.5), np.percentile(counts, 99.5)]

    n_bins = int(len(counts)/35)
    if exp_count < 30:
        n_bins = int(n_bins / 4)

    if len(counts) < 900:
        n_bins = int(n_bins*3)

    hist, bin_edges = np.histogram(counts, bins=n_bins, range=(lb, ub))
    hist_g, bin_edges_g = np.histogram(counts_g, bins=n_bins, range=(lb, ub))
    hist_c, bin_edges_c = np.histogram(counts_c, bins=n_bins, range=(lb, ub))
    hist_a, bin_edges_a = np.histogram(counts_a, bins=n_bins, range=(lb, ub))
    hist_t, bin_edges_t = np.histogram(counts_t, bins=n_bins, range=(lb, ub))

    hist_max = np.max([np.max(hist_g), np.max(hist_c), np.max(hist_a), np.max(hist_t)])

    fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    # plt.bar(bin_edges[:-1], hist, width=(bin_edges[-1]-bin_edges[0])/n_bins, alpha=0.7, align='edge',
    #         color='none', edgecolor='black')
    axs[0].bar(bin_edges_g[:-1], hist_g, width=(bin_edges_g[-1]-bin_edges_g[0])/n_bins,
             alpha=0.9, align='edge', color='green', label='G')
    axs[1].bar(bin_edges_c[:-1], hist_c, width=(bin_edges[-1] - bin_edges[0]) / n_bins,
            alpha=0.9, align='edge', color='orange', label='C')
    axs[2].bar(bin_edges_a[:-1], hist_a, width=(bin_edges[-1] - bin_edges[0]) / n_bins,
            alpha=0.9, align='edge', color='black', label='A')
    axs[3].bar(bin_edges_t[:-1], hist_t, width=(bin_edges[-1] - bin_edges[0]) / n_bins,
             alpha=0.9, align='edge', color='purple', label='T')
    for ax in axs:
        ax.legend()
        ax.set_ylim(0, hist_max)
    plt.xlim((lb, ub))
    axs[0].set_title(nt_x + ' --> ' + nt_y + ' (paired)')
    plt.xlabel('log(counts)')
    plt.savefig(nt_x + nt_y + '_cont_log_paired_'+cont_dir+'.png')
    plt.show()

    # Extract counts from dataframe
    counts = np.log(df['actual_count'].values)

    # Extract counts for paired/unpaired sites dependent on context
    counts_g = counts[(df['unpaired'].values == 1) * (df[cont_dir + '_context'].values == 'G')]
    counts_c = counts[(df['unpaired'].values == 1) * (df[cont_dir + '_context'].values == 'C')]
    counts_a = counts[(df['unpaired'].values == 1) * (df[cont_dir + '_context'].values == 'A')]
    counts_t = counts[(df['unpaired'].values == 1) * (df[cont_dir + '_context'].values == 'T')]

    [log_min, log_max] = [np.min(counts), np.max(counts)]

    [lb, ub] = [np.percentile(counts, 0.5), np.percentile(counts, 99.5)]

    n_bins = int(len(counts) / 35)
    if exp_count < 30:
        n_bins = int(n_bins / 4)

    if len(counts) < 900:
        n_bins = int(n_bins * 3)

    hist, bin_edges = np.histogram(counts, bins=n_bins, range=(lb, ub))
    hist_g, bin_edges_g = np.histogram(counts_g, bins=n_bins, range=(lb, ub))
    hist_c, bin_edges_c = np.histogram(counts_c, bins=n_bins, range=(lb, ub))
    hist_a, bin_edges_a = np.histogram(counts_a, bins=n_bins, range=(lb, ub))
    hist_t, bin_edges_t = np.histogram(counts_t, bins=n_bins, range=(lb, ub))

    hist_max = np.max([np.max(hist_g), np.max(hist_c), np.max(hist_a), np.max(hist_t)])

    fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    # plt.bar(bin_edges[:-1], hist, width=(bin_edges[-1]-bin_edges[0])/n_bins, alpha=0.7, align='edge',
    #         color='none', edgecolor='black')
    axs[0].bar(bin_edges_g[:-1], hist_g, width=(bin_edges_g[-1] - bin_edges_g[0]) / n_bins,
               alpha=0.9, align='edge', color='green', label='G')
    axs[1].bar(bin_edges_c[:-1], hist_c, width=(bin_edges[-1] - bin_edges[0]) / n_bins,
               alpha=0.9, align='edge', color='orange', label='C')
    axs[2].bar(bin_edges_a[:-1], hist_a, width=(bin_edges[-1] - bin_edges[0]) / n_bins,
               alpha=0.9, align='edge', color='black', label='A')
    axs[3].bar(bin_edges_t[:-1], hist_t, width=(bin_edges[-1] - bin_edges[0]) / n_bins,
               alpha=0.9, align='edge', color='purple', label='T')
    for ax in axs:
        ax.legend()
        ax.set_ylim(0, hist_max)
    plt.xlim((lb, ub))
    axs[0].set_title(nt_x + ' --> ' + nt_y + ' (unpaired)')
    plt.xlabel('log(counts)')
    plt.savefig(nt_x + nt_y + '_cont_log_unpaired_'+cont_dir+'.png')
    plt.show()

    return 0


if __name__ == '__main__':

    for nt_1 in ['T']:

        for nt_2 in ['G']:

            if nt_1 != nt_2:

                # Define mutation type X --> Y, cell type, and pseudo_count
                [nt_x, nt_y, cell_type, pseudo_count] = [nt_1, nt_2, 'Huh7', 0.5]

                # Load data
                df, exp_count = load_data(nt_x=nt_x, nt_y=nt_y, cell_type=cell_type, pseudo_count=pseudo_count)

                # Plot mutation counts depending on context
                for dir in ['back', 'forw']:
                    context_plot(df, nt_x, nt_y, exp_count, pseudo_count, cont_dir=dir)
                    #context_plot_log(df, nt_x, nt_y, exp_count, pseudo_count, cont_dir=dir)