.. _Metrics:

==============================
Distribution Metrics
==============================

These are metrics utilized to compare 2 distributions. The LOC Curve,
which utilizes the length of the curve to quantify separation,
is the generalized case of the traditional ROC curve, which
utilized the area under the curve using numpy.trapz.

.. note:: **Note: Both functions require 1-dimensional arrays as input.**
    This is simply to make the function completely generalizeable. One can
    make any N-dimensional function "1-dimensional" by calling :py:func:`numpy.ndarray.ravel`.
    The functions will sort the bins and handle the rest.


.. autofunction:: MiLoMerge.ROC_curve

.. autofunction:: MiLoMerge.LOC_curve
