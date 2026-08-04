#!/usr/bin/env python3
"""
Finds variable-size block structure of a sparse CSR matrix by growing a
reach "frontier" row by row. A block boundary is declared once the
frontier stops advancing -- i.e. once no row in the current unresolved
range points to a column beyond what's already been reached.

Requires a.indices to be sorted ascending per row (sort_indices() is
called before use).
"""

import sys
import numpy as np
import scipy.sparse as sp


def find_block_slices(a: sp.csr_matrix):
    n = a.shape[0]
    block_slices = []
    visited_max = -1
    frontier_max = 0

    while frontier_max > visited_max:
        start = visited_max + 1
        stop = frontier_max + 1
        block_slices.append(slice(start, stop))

        next_max = frontier_max
        for node in range(start, stop):
            row_start = a.indptr[node]
            row_stop = a.indptr[node + 1]
            if row_stop > row_start:
                row_max = int(a.indices[row_stop - 1])
                next_max = max(next_max, row_max)

        visited_max = frontier_max
        frontier_max = min(next_max, n - 1)

    if len(block_slices) > 1:
        block_slices[1] = slice(block_slices[0].start, block_slices[1].stop)
        block_slices.pop(0)

    return block_slices


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python block_slices.py <matrix.npz>")
        sys.exit(1)

    d = np.load(sys.argv[1])
    A = sp.csr_matrix((d['data'], d['indices'], d['indptr']), shape=tuple(d['shape']))
    A.sort_indices()

    slices = find_block_slices(A)
    sizes = [s.stop - s.start for s in slices]

    print(f"matrix shape = {A.shape}, nnz = {A.nnz}")
    print(f"number of blocks found = {len(slices)}")
    print(f"block sizes = {sizes}")