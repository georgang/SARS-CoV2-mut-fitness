import pandas as pd


def extract_segments(arr):

    segments = []
    start_index = 0
    current_value = arr[0]

    for i in range(1, len(arr)):
        if arr[i] != current_value:
            end_index = i - 1
            segments.append((start_index + 1, end_index + 1, current_value))
            start_index = i
            current_value = arr[i]

    segments.append((start_index, len(arr) - 1, current_value))

    return segments


if __name__ == '__main__':

    CLADE = '21J'  # Does it matter which clade I take?

    # Get dataset for the selected clade
    df = pd.read_csv(f'counts/counts_all_{CLADE}.csv')

    # Get a list of length 29903 with the gene assignment of every site
    genes = df['gene'].values[::3]

    # Extract boundaries of genes
    segments = extract_segments(genes)

    # Print boundaries of genes
    for start, end, value in segments:
        print(f"Start: {start}, End: {end}, Value: {value}")
