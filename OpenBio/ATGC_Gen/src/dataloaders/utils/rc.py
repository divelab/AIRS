"""Utility functions for reverse complementing DNA sequences.

"""

from random import random

STRING_COMPLEMENT_MAP = {
    "A": "T", "C": "G", "G": "C", "T": "A", "a": "t", "c": "g", "g": "c", "t": "a",
    "N": "N", "n": "n",
}

def coin_flip(p=0.5):
    """Flip a (potentially weighted) coin."""
    return random() > p


def string_reverse_complement(seq):
    """Reverse complement a DNA sequence."""
    rev_comp = ""
    for base in seq[::-1]:
        if base in STRING_COMPLEMENT_MAP:
            rev_comp += STRING_COMPLEMENT_MAP[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp
