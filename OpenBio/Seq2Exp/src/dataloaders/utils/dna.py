import numpy as np


def one_hot_to_dna_str(one_hot_seq):
    bases = np.array(['A', 'C', 'G', 'T', 'N'])

    max_indices = np.argmax(one_hot_seq, axis=1)

    is_invalid = np.sum(one_hot_seq, axis=1) != 1
    max_indices[is_invalid] = 4

    dna_seq = bases[max_indices]

    return ''.join(dna_seq)


def dna_str_to_one_hot(dna_sequence):
    mapping = {
        'A': [True, False, False, False],
        'C': [False, True, False, False],
        'G': [False, False, True, False],
        'T': [False, False, False, True],
        'N': [False, False, False, False]
    }

    one_hot_encoded = np.array([mapping[nucleotide] for nucleotide in dna_sequence], dtype=bool)
    return one_hot_encoded
