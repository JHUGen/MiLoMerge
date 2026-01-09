import warnings
import os
import numpy as np
import h5py


def __load_file_local(fname, key):
    """Simple function to load and return an h5py file

    Parameters
    ----------
    fname : str
        The filename to load
    key : str
        The key to load for the h5py file

    Returns
    -------
    numpy.ndarray
        Returns a copy of the array that is being accessed through h5py
    """
    f = h5py.File(fname, "r")
    return f[key][:]


def __load_file_nonlocal(fname_tracker, fname_bins, key):
    """Simple function to load and return an h5py file
    alongside its corresponding "physical bins" file

    Parameters
    ----------
    fname_tracker : str
        The filename of the tracker (h5py file)
    fname_bins : str
        The filename of the physical bins file
    key : str
        The key to load for the h5py file

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Returns a copy of the array that is being accessed through h5py
        alongside a copy of the physical bins
    """
    f = h5py.File(fname_tracker, "r")
    bin_mapping = f[key][:]

    physical_bins = np.load(fname_bins, allow_pickle=True)

    return bin_mapping, physical_bins


def place_event_nonlocal(N, *observable, file_prefix, verbose=False):
    """This function takes in one N-dimensional observable from data
    and utilizes the mapping and saved physical bins
    from MiLoMerge.MergerNonlocal to output where
    that event would go in the final nonlocal binning.

    Parameters
    ----------
    N : int
        The number of bins in the mapping to use
    observable : float
        args input where one inputs observables
        in the same order as the input binning to MiLoMerge
    file_prefix : str
        The entirety of the filepath before _tracker.hdf5 or
        "_physical_bins.npy". This argument should
        be the same as `f"{file_path} + {file_prefix}"`,
        where `file_path` and `file_prefix` are the inputs
        given to MiLoMerge.MergerNonlocal.
    verbose : bool, optional
        Whether additional print statements
        are turned on, by default False

    Returns
    -------
    int
        The index of where the event is placed in the final binning.

    Raises
    ------
    FileNotFoundError
        If the prefix is not suitable to find the appropriate
        tracker.hdf5 file, raise an error.
    FileNotFoundError
        If the prefix is not suitable to find the appropriate
        physical_bins.npy file, raise an error.
    ValueError
        If any observable is outside of the provided bins, raise
        an error
    ValueError
        If the dimensions of the observable and 
        the dimensions of the bins are not compatible, raise
        an error
    """
    if not os.path.exists(f"{file_prefix}_tracker.hdf5"):
        raise FileNotFoundError(f"{file_prefix}_tracker.hdf5 does not exist!")
    if not os.path.exists(f"{file_prefix}_physical_bins.npy"):
        raise FileNotFoundError(f"{file_prefix}_physical_bins.npy does not exist!")

    fname_tracker = f"{file_prefix}_tracker.hdf5"
    fname_bins = f"{file_prefix}_physical_bins.npy"
    bin_mapping, physical_bins = __load_file_nonlocal(fname_tracker, fname_bins, str(N))

    observable = np.array(observable)

    if not (isinstance(physical_bins[0], int) or isinstance(physical_bins[0], float)):
        subarray_lengths = np.array([len(b) for b in physical_bins])
    else:
        subarray_lengths = np.array([len(physical_bins)])
    if len(physical_bins.shape) > 1 and len(observable) > 1:
        n_observables = physical_bins.shape[0]
        n_physical_bins = physical_bins.shape[1]

        nonzero_rolled = np.zeros(n_observables, dtype=np.uint64)
        for i in np.arange(n_observables):
            nonzero_rolled[i] = np.searchsorted(physical_bins[i], observable[i]) - 1

        if np.any(nonzero_rolled < 0) or np.any(nonzero_rolled > n_physical_bins - 1):
            raise ValueError(f"observables are outside of the provided phase space!")

        if verbose:
            print("original index of:", nonzero_rolled)
            print("This places your point in the range:")
            for i in range(physical_bins.shape[0]):
                print(
                    "[",
                    physical_bins[i][nonzero_rolled[i]],
                    ",",
                    physical_bins[i][int(nonzero_rolled[i] + 1)],
                    "]",
                )

        unrolled_index = (
            np.power(
                n_physical_bins - 1, np.arange(n_observables - 1, -1, -1, np.int16)
            )
            * nonzero_rolled
        ).sum()

    elif any([subarray_lengths[0] != b for b in subarray_lengths]) and len(
        subarray_lengths
    ) == len(observable):
        n_observables = len(observable)
        n_physical_bins = np.array(subarray_lengths)  # this is now per dimension

        nonzero_rolled = np.zeros(n_observables, dtype=np.uint64)
        for i in np.arange(n_observables):
            nonzero_rolled[i] = np.searchsorted(physical_bins[i], observable[i]) - 1
            if nonzero_rolled[i] < 0 or nonzero_rolled[i] >= n_physical_bins[i]:
                raise ValueError(
                    f"Observable of index {i} is outside of the provided phase space!"
                )

            if verbose:
                print("original index of:", nonzero_rolled[i])
                print("This places your point in the range:")
                print(
                    "[",
                    physical_bins[i][nonzero_rolled[i]],
                    ",",
                    physical_bins[i][int(nonzero_rolled[i] + 1)],
                    "]",
                )

        unrolled_index = 0
        multiplier = 1
        for i in reversed(range(len(n_physical_bins))):
            unrolled_index += nonzero_rolled[i] * multiplier
            if i > 0:
                multiplier *= subarray_lengths[i] - 1

    elif len(physical_bins.shape) > 1 or len(observable) > 1:
        raise ValueError(
            f"Shapes are incompatible\n"
            f" shape of bins is {len(physical_bins.shape)} and"
            f" length of observables is {len(observable)}"
        )
    else:
        if len(observable) > 1:
            raise ValueError

        unrolled_index = np.searchsorted(physical_bins, observable) - 1

    unrolled_index = int(unrolled_index)
    try:
        mapped_index = bin_mapping[unrolled_index]
    except IndexError as e:
        raise ValueError(f"{observable} is outside of the provided phase space!") from e

    return mapped_index


def place_array_nonlocal(N, observables, file_prefix="", verbose=False):
    """This function takes in an array of N-dimensional observables from data
    and utilizes the mapping and saved physical bins
    from MiLoMerge.MergerNonlocal to output where
    that event would go in the final nonlocal binning. The array
    version of place_event_nonlocal.

    Parameters
    ----------
    N : int
        The number of bins in the mapping to use
    observables : numpy.ndarray[float]
        Array should be of shape (#events, #observables). Contains
        all the observables to be binned.
    file_prefix : str
        The entirety of the filepath before _tracker.hdf5 or
        "_physical_bins.npy". This argument should
        be the same as `f"{file_path} + {file_prefix}"`,
        where `file_path` and `file_prefix` are the inputs
        given to MiLoMerge.MergerNonlocal.
    verbose : bool, optional
        Whether additional print statements
        are turned on, by default False

    Returns
    -------
    numpy.ndarray[int]
        A 1-d array of indices
        for where the events are placed in the final binning.

    Raises
    ------
    FileNotFoundError
        If the prefix is not suitable to find the appropriate
        tracker.hdf5 file, raise an error.
    FileNotFoundError
        If the prefix is not suitable to find the appropriate
        physical_bins.npy file, raise an error.
    ValueError
        If any observable is outside of the provided bins, raise
        an error
    ValueError
        If the dimensions of the observable and 
        the dimensions of the bins are not compatible, raise
        an error
    KeyError
        If any observable is outside the provided bins,
        raise an error
    """

    if not os.path.exists(f"{file_prefix}_tracker.hdf5"):
        raise FileNotFoundError(f"{file_prefix}_tracker.hdf5 does not exist!")
    if not os.path.exists(f"{file_prefix}_physical_bins.npy"):
        raise FileNotFoundError(f"{file_prefix}_physical_bins.npy does not exist!")

    fname_tracker = f"{file_prefix}_tracker.hdf5"
    fname_bins = f"{file_prefix}_physical_bins.npy"
    bin_mapping, physical_bins = __load_file_nonlocal(fname_tracker, fname_bins, str(N))
    observables_stacked = np.array(observables)

    if not (isinstance(physical_bins[0], int) or isinstance(physical_bins[0], float)):
        subarray_lengths = np.array([len(b) for b in physical_bins])
    else:
        subarray_lengths = np.array([len(physical_bins)])
    if physical_bins.ndim > 1:
        if len(observables_stacked[0]) != len(physical_bins):
            raise ValueError(
                f"Number of observables {len(observables_stacked[0])} != Number of bin dimensions {len(physical_bins)}"
            )
        n_physical_bins = physical_bins.shape[1]

        n_datapoints, n_observables = observables_stacked.shape
        nonzero_rolled = np.zeros((n_datapoints, n_observables), dtype=np.uint64)
        for i in range(n_observables):
            left_edge_mask = observables_stacked[:, i] == physical_bins[i][0]
            nonzero_rolled[:, i][~left_edge_mask] = (
                np.searchsorted(
                    physical_bins[i], observables_stacked[:, i][~left_edge_mask]
                )
                - 1
            )
        if verbose:
            print("Original indices")
            print(nonzero_rolled)

        unrolled_index = (
            np.power(
                n_physical_bins - 1, np.arange(n_observables - 1, -1, -1, np.int64)
            )
            * nonzero_rolled
        ).sum(axis=1)
        unrolled_index = unrolled_index.astype(int)

    elif np.any(subarray_lengths != subarray_lengths[0]):
        if len(observables_stacked[0]) != len(physical_bins):
            raise ValueError(
                f"Number of observables {len(observables_stacked[0])} != Number of bin dimensions {len(physical_bins)}"
            )
        n_physical_bins = subarray_lengths
        n_datapoints, n_observables = observables_stacked.shape
        nonzero_rolled = np.zeros((n_datapoints, n_observables), dtype=np.uint64)
        for i in range(n_observables):
            nonzero_rolled[:, i] = (
                np.searchsorted(physical_bins[i], observables_stacked[:, i]) - 1
            )
        if verbose:
            print("Original indices")
            print(nonzero_rolled)

        unrolled_index = np.zeros(n_datapoints, dtype=np.uint64)
        multiplier = 1
        for i in reversed(range(len(n_physical_bins))):
            unrolled_index += nonzero_rolled[:, i] * multiplier
            if i > 0:
                multiplier *= subarray_lengths[i] - 1

    else:
        if observables_stacked.ndim != physical_bins.ndim:
            raise ValueError(
                f"Number of observables {observables_stacked.ndim} != Number of bin dimensions {physical_bins.ndim}"
            )
        n_physical_bins = len(physical_bins)
        nonzero_rolled = np.searchsorted(physical_bins, observables_stacked) - 1
        unrolled_index = nonzero_rolled

    failed_events = unrolled_index > len(bin_mapping)
    if np.any(failed_events):
        print("The following events have indices that are too large:")
        for i, j in zip(nonzero_rolled[failed_events], unrolled_index[failed_events]):
            print(f"{i} = {j}")
        raise KeyError(
            "Please check your phasespace to ensure it is within your original binning!"
        )

    return bin_mapping[unrolled_index].ravel()


def place_local(N, observable_array, file_prefix, verbose=False):
    """Places 1-dimensional data into the respective bins for
    a given bin number. Equivalent to running numpy.histogram
    for the given N+1 bin edges stored.

    Parameters
    ----------
    N : int
        The number of bins that are desired
    observable_array : numpy.ndarray
        A 1-d array of datapoints to be binned with N bins
    file_prefix : str, optional
        The entirety of the filepath before _tracker.hdf5 or
        "_physical_bins.npy". This argument should
        be the same as `f"{file_path} + {file_prefix}"`,
        where `file_path` and `file_prefix` are the inputs
        given to MiLoMerge.MergerNonlocal.
    verbose : bool, optional
        Whether additional print statements
        are turned on, by default False, by default False

    Returns
    -------
    numpy.ndarray[int]
        A 1-d array of indices
        for where the events are placed in the final binning.

    Raises
    ------
    FileNotFoundError
        If the prefix is not suitable to find the appropriate
        tracker.hdf5 file, raise an error.
    """
    if not os.path.exists(f"{file_prefix}_tracker.hdf5"):
        raise FileNotFoundError(f"{file_prefix}_tracker.hdf5 does not exist!")

    fname_tracker = f"{file_prefix}_tracker.hdf5"
    bin_mapping = __load_file_local(fname_tracker, str(N))

    if verbose:
        print(f"Using file {os.path.abspath(fname_tracker)}")
        print(np.array(bin_mapping))

    placements = np.searchsorted(bin_mapping, observable_array) - 1

    if np.any((placements < 0) | (placements >= len(bin_mapping))):
        warnings.warn(
            "Some items placed out of bounds! Please check your phasespace to ensure it is within your original binning!"
        )

    return np.bincount(placements, minlength=N), bin_mapping
