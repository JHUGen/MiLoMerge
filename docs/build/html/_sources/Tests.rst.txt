.. _Tests:

==============================
Test Cases
==============================

The :py:func:`test_2_case` tests in :ref:`Localtest` and :ref:`NonlocalTest`
showcase the fact that the brute force case and the MiLoMerge package produce the same
result.

.. _LocalTest:

Local Usage
++++++++++++

.. literalinclude :: ../../tests/test_merge_local.py
   :language: python

.. _NonlocalTest:

Nonlocal Usage
+++++++++++++++

.. literalinclude :: ../../tests/test_merge_nonlocal.py
   :language: python

Post-merging Bin Placement
++++++++++++++++++++++++++++

.. literalinclude :: ../../tests/test_placement.py
   :language: python
