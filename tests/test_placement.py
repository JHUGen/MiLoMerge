import numpy as np
import pytest
import MiLoMerge
import helpers

def test_local_placement():
    bin_centers = (np.arange(1, 31) + np.arange(30))/2

    h1 = np.zeros(30)
    h1[0] = 5
    h1[1] = 1
    h1_data = [bin_centers[0]]*int(h1[0]) + [bin_centers[1]]*int(h1[1])

    h2 = np.zeros(30)
    h2[2] = 5
    h2[3] = 2
    h2_data = [bin_centers[2]]*int(h2[2]) + [bin_centers[3]]*int(h2[3])
    
    h3 = np.zeros(30)
    h3[4] = 5
    h3[5] = 3
    h3_data = [bin_centers[4]]*int(h3[4]) + [bin_centers[5]]*int(h3[5])

    merger = MiLoMerge.MergerLocal(
        range(31), #bin edges
        h1,
        h2,
        h3, 
        map_at=(3,)
    )
    new_edges, new_counts = merger.run(3, return_counts=True)
    for i, arr in enumerate((h1_data, h2_data, h3_data)):
        c, b = MiLoMerge.place_local(3, arr, "./", verbose=False)
        assert np.array_equal(c, new_counts[i])
        assert np.array_equal(b, new_edges)


def test_nonlocal_placement_1d():
    bin_centers = (np.arange(1, 31) + np.arange(30))/2
    h1 = np.zeros(30)
    h1[0] = 5
    h1[3] = 1
    h1_data = [bin_centers[0]]*int(h1[0]) + [bin_centers[3]]*int(h1[3])

    h2 = np.zeros(30)
    h2[1] = 5
    h2[4] = 2
    h2_data = [bin_centers[1]]*int(h2[1]) + [bin_centers[4]]*int(h2[4])
    
    h3 = np.zeros(30)
    h3[2] = 5
    h3[5] = 3
    h3_data = [bin_centers[2]]*int(h3[2]) + [bin_centers[5]]*int(h3[5])

    merger = MiLoMerge.MergerNonlocal(
        range(31), #bin edges
        h1,
        h2,
        h3,
        map_at=(3,)
    )
    new_counts = merger.run(3)
    for i, arr in enumerate((h1_data, h2_data, h3_data)):
        temp_counts = np.zeros(3)
        placements = MiLoMerge.place_array_nonlocal(3, arr, "./")
        temp_counts, _ = np.histogram(placements, range(3+1))
        assert np.array_equal(temp_counts, new_counts[i])