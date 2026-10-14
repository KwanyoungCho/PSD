"""Reuse already validated CPU topology for draft glue inputs."""
import numpy as np


def context_topology(parents):
    n = len(parents)
    depth = [0] * n
    ancestors = [[] for _ in range(n)]
    rows = np.zeros((n + 1, n + 1), dtype=np.uint8)
    rows[0, 0] = 1
    for j, parent in enumerate(parents):
        parent = int(parent)
        if parent < -1 or parent >= j:
            raise ValueError(f'Invalid tree parent {parent} for node {j}')
        if parent >= 0:
            depth[j] = depth[parent] + 1
            ancestors[j] = ancestors[parent] + [parent]
        rows[j+1, 0] = rows[j+1, j+1] = 1
        for a in ancestors[j]: rows[j+1, a+1] = 1
    return depth, ancestors, rows
