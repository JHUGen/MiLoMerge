import numpy as np
import h5py
import warnings
import os

def load_file_local(fname, key):
    f = h5py.File(fname, 'r')
    return f[key][:]

def load_file_nonlocal(fname_tracker, fname_bins, key):
    f = h5py.File(fname_tracker, 'r')
    bin_mapping = f[key][:]

    physical_bins = np.load(fname_bins)

    return bin_mapping, physical_bins

def place_event_nonlocal(N, *observable, file_prefix="", verbose=False):

    fname_tracker = f".{file_prefix}_tracker.hdf5"
    fname_bins = f".{file_prefix}_physical_bins.npy"
    bin_mapping, physical_bins = load_file_nonlocal(fname_tracker, fname_bins, str(N))

    observable = np.array(observable)

    if len(physical_bins.shape) > 1 and len(observable) > 1:
        n_observables = physical_bins.shape[0]
        n_physical_bins = physical_bins.shape[1]

        nonzero_rolled = np.zeros(n_observables, dtype=np.uint16)
        for i in np.arange(n_observables):
            nonzero_rolled[i] = np.searchsorted(physical_bins[i], observable[i]) - 1

        if np.any(nonzero_rolled < 0) or np.any(nonzero_rolled > n_physical_bins - 1):
            raise ValueError(f"{observable} is outside of the provided phase space!")

        if verbose:
            print('original index of:', nonzero_rolled)
            print('This places your point in the range:')
            for i in range(physical_bins.shape[0]):
                print('[', physical_bins[i][nonzero_rolled[i]], ',', physical_bins[i][int(nonzero_rolled[i]+1)], ']')

        unrolled_index = (np.power(n_physical_bins - 1, np.arange(n_observables - 1,-1,-1, np.int16))*nonzero_rolled).sum()

    elif len(physical_bins.shape) > 1 or len(observable) > 1:
        raise ValueError("Shapes are incompatible")
    else:
        if len(observable) > 1:
            raise ValueError

        unrolled_index = np.searchsorted(physical_bins, observable) - 1
    
    try:
        mapped_index = bin_mapping[unrolled_index][0]
    except IndexError:
        raise ValueError(f"{observable} is outside of the provided phase space!")
    except:
        raise

    return mapped_index

def place_array_nonlocal(N, observables, file_prefix="", verbose=False):

    fname_tracker = f".{file_prefix}_tracker.hdf5"
    fname_bins = f".{file_prefix}_physical_bins.npy"
    bin_mapping, physical_bins = load_file_nonlocal(fname_tracker, fname_bins, str(N))
    observables_stacked = np.array(observables)
    
    if physical_bins.ndim > 1:
        if len(observables_stacked[0]) != len(physical_bins):
            raise ValueError(f"Number of observables {len(observables_stacked[0])} != Number of bin dimensions {len(physical_bins)}")
        n_physical_bins = physical_bins.shape[1]
        
        n_datapoints, n_observables = observables_stacked.shape
        nonzero_rolled = np.zeros((n_datapoints, n_observables))
        for i in range(n_observables):
            nonzero_rolled[:, i] = np.searchsorted(physical_bins[i], observables_stacked[:, i])
        if verbose:
            print("Original indices")
            print(nonzero_rolled)
    
        unrolled_index = (np.power(n_observables - 1, np.arange(n_observables - 1,-1,-1, np.int16))*nonzero_rolled).sum(axis=1)
    
    else:
        if observables_stacked.ndim != physical_bins.ndim:
            raise ValueError(f"Number of observables {1} != Number of bin dimensions {1}")
        n_physical_bins = len(physical_bins)
        nonzero_rolled = np.searchsorted(physical_bins, observables_stacked) - 1
        unrolled_index = nonzero_rolled

    test_arr = np.logical_or(nonzero_rolled < 0, nonzero_rolled > n_physical_bins - 1)
    if np.any(test_arr):
        raise ValueError(f"observables at indices {np.nonzero(test_arr)} is outside of the provided phase space!")
    
    return bin_mapping[unrolled_index].ravel()

def place_local(N, observable_array, file_prefix="", verbose=False):
    fname = f".{file_prefix}_tracker.hdf5"
    bin_mapping = load_file_local(fname, str(N))

    if verbose:
        print(f"Using file {os.path.abspath(fname)}")
        print(np.array(bin_mapping))

    placements = np.searchsorted(bin_mapping, observable_array) - 1

    if np.any((placements < 0) or (placements == len(bin_mapping)) ):
        warnings.warn("Some items placed out of bounds! Consider having an overflow or underflow bin!")

    return placements