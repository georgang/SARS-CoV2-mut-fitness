'''
Create map between row in matrix and mutation type, context, and pairing state
'''

row_keys = []

for anc_nt in ['A', 'C', 'G', 'T']:

    for mut_nt in ['A', 'C', 'G', 'T']:

        if anc_nt != mut_nt:

            for x in ['p', 'up']:

                for left_nt in ['A', 'C', 'G', 'T']:

                    for right_nt in ['A', 'C', 'G', 'T']:

                        row_keys.append(left_nt + anc_nt + right_nt + '_' + mut_nt + '_' + x)

print(row_keys)
