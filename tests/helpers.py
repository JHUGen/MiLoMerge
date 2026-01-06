import itertools
import numpy as np
import MiLoMerge

def merge(h1, h2, i1, i2):
    h1_temp = h1.copy()
    h1_temp[i1] += h1_temp[i2]
    h1_temp = np.delete(h1_temp, i2)

    h2_temp = h2.copy()
    h2_temp[i1] += h2_temp[i2]
    h2_temp = np.delete(h2_temp, i2)
    return h1_temp, h2_temp

def brute_force(h1, h2, target, local=False):
    if len(h1) == target:
        return h1, h2
    merge_possibilities = list(itertools.combinations(range(len(h1)), 2))

    cur_max = 0
    cur_indices = [None, None]
    for i1, i2 in merge_possibilities:
        if local and (i1 != i2 - 1) and (i1 != i2 + 1):
            continue
    
        h1_temp, h2_temp = merge(h1, h2, i1, i2)
        
        score = MiLoMerge.ROC_curve(h1_temp, h2_temp)[2]
        if score > cur_max:
            cur_max = score
            cur_indices = (i1, i2)
    
    h1, h2 = merge(h1, h2, *cur_indices)
    return brute_force(h1, h2, target)
    