import numpy as np
import pandas as pd

df = pd.read_csv('counts/counts_all.csv')
df = df[(df['clade'] == '20A')]
arr = df['gene'][::3].values


def find_string_occurrences(arr):
    result = []

    first_occurrence = 1
    current_string = arr[0]

    for i in range(1, len(arr)):
        if arr[i] != current_string:
            result.append((first_occurrence, i, current_string))
            current_string = arr[i]
            first_occurrence = i + 1

    result.append((first_occurrence, len(arr), current_string))  # Capture the last segment

    return result


# Example usage:
output = find_string_occurrences(arr)
print(output)
