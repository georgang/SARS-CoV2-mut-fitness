import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_segments(arr):
    if len(arr) == 0:
        return []

    segments = []
    start_index = 0
    current_value = arr[0]

    for i in range(1, len(arr)):
        if arr[i] != current_value:
            end_index = i - 1
            segments.append((start_index, end_index, current_value))
            start_index = i
            current_value = arr[i]

    # Add the last segment
    segments.append((start_index, len(arr) - 1, current_value))

    return segments


df = pd.read_csv('../counts/counts_all.csv')

df = df[(df['clade'] == '21J')]


genes = df['gene'].values[::3]

segments = extract_segments(genes)

for start, end, value in segments:
    print(f"Start: {start}, End: {end}, Value: {value}")
