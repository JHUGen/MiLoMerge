.. _Examples:

==============================
Basic Examples
==============================

While not a formal tutorial, this page code snippets that 
one may find useful for certain use cases. 


Generating a Toy Dataset and Plotting Separability as a Function of Bin Number
--------------------------------------------------------------------------------

.. warning:: 
    This code snippet is best used for either local merging
    or for nonlocal merging where the number of bins is 
    :math:`~\leq 2500`. Any longer then that and
    it would take too long, and we instead recommend
    that one utilizes the map_at functionality
    to instead map the performance at a variety
    of different selected bin values.

Since both the merger classes in :ref:`Mergers` 
save the number of bins from the run method, one can
keep calling the run method.

.. code-block:: python
    :name: perfPerBinExample

    import matplotlib.pyplot as plt
    import numpy as np
    import MiLoMerge

    signal = np.random.normal(loc=600, scale=20, size=5000) #some random signal centered at 600
    background = 175 + 1000*np.random.exponential(0.2, 100_000) #some random falling background
    binning = np.arange(175, 1001, 25)

    background_counts, _ = np.histogram(background, binning)
    background_counts = background_counts*1000/background_counts.sum()
    background_counts = background_counts.astype(int)

    signal_counts, _ = np.histogram(signal, binning)
    signal_counts = signal_counts*50/signal_counts.sum()
    signal_counts = signal_counts.astype(int)

    # here we are starting from 25 bins
    merger = MiLoMerge.MergerLocal(binning, h1, h2)

    n_bins = np.arange(25, 2, -1, dtype=np.int32)
    scores = np.zeros_like(n_bins, dtype=np.float64)

    for i, n in enumerate(n_bins):
        # We can unpack the 2-d bin counts for our own use
        (h1_temp, h2_temp), _ = merger.run(n, return_counts=True)
        scores[i] = MiLoMerge.LOC_score(h1_temp, h2_temp)

    plt.plot(n_bins, scores)


Using the ROC and LOC curves for N-d distributions
---------------------------------------------------

The ROC and LOC curve functions documented in the :ref:`Metrics` 
section take 1-dimensional arrays. This does not mean that
they have to be 1-dimensional distributions! Since
the curves are ordered by ratio, one can simply
call :py:func:`numpy.ndarray.ravel` to unroll the histogram
into an equivalent 1-dimensional distribution.

.. code-block:: python
    :name: ROC_LOC_example

    import matplotlib.pyplot as plt
    import numpy as np
    import MiLoMerge

    # Imagine that h1 and h2 are n-dimensional NumPy histograms
    # of some arbitrary variable

    #Using the LOC curve
    TPR, FPR, score = MiLoMerge.LOC_curve(h1.ravel(), h2.ravel())

    #Using the ROC curve
    TPR, FPR, score = MiLoMerge.ROC_curve(h1.ravel(), h2.ravel())


