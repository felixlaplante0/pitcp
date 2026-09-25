PIT-CP
======

**pitcp** is a Python package for conformal prediction with probability integral
transform (PIT) pivotal scores. It fits a conditional density estimator to any
scalar nonconformity score, maps scores to PIT values, and calibrates them at any
confidence level.

.. code-block:: bash

   pip install pitcp

See :doc:`getting-started` for a first example, or try the
`interactive playground <https://pitcp-app.streamlit.app/>`_. The package is
available on `PyPI <https://pypi.org/project/pitcp/>`_, and the method is described in
the `paper <https://doi.org/10.48550/arXiv.2605.25852>`_.

Why PIT-CP?
-----------

Split conformal prediction guarantees coverage on average over inputs, which can
hide large local errors: regions are too wide where noise is low and too narrow where
it is high. ``PITCP`` learns the conditional distribution of the score and maps it to
a pivotal PIT value before calibration, so thresholds follow the data. One fitted
density serves every confidence level.

Quick example
-------------

.. code-block:: python

   import torch
   import zuko
   from pitcp import PITCP

   # Any scalar nonconformity score s(x, y)
   def score(x, y):
       return y.abs()

   flow = zuko.flows.NSF(features=1, context=1, hidden_features=(32, 32))
   optimizer = torch.optim.Adam(flow.parameters(), lr=1e-2)

   model = PITCP(flow, optimizer, n_epochs=10, batch_size=128)
   model.fit(X_train, score(X_train, y_train))
   model.conformalize(X_cal, score(X_cal, y_cal))

   # Score thresholds at several levels, without refitting
   limits = model.predict(X_test, confidence_level=[0.7, 0.8, 0.9])

The :doc:`getting-started` page explains the train / calibrate / predict split, and
the :doc:`tutorial` walks through a full example.

Choose a method
---------------

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: PITCP
      :link: methods
      :link-type: doc

      Conditional score model. Best starting point for adaptive regions.

   .. grid-item-card:: SCP
      :link: methods
      :link-type: doc

      The smallest baseline: one calibrated score threshold for every input.

   .. grid-item-card:: CQR
      :link: methods
      :link-type: doc

      Learns response quantiles and must be refitted when the confidence level changes.

   .. grid-item-card:: HPD and CONTRA
      :link: methods
      :link-type: doc

      Density-level and latent-space regions for richer conditional distributions.

Citation
--------

If you use PIT-CP, please cite:

.. code-block:: bibtex

   @article{laplante2026pitcp,
     title   = {A Post-Processing Conformal Prediction Approach for Conditional
                Coverage via Pivotal Scores},
     author  = {Laplante, F{\'e}lix},
     journal = {arXiv preprint arXiv:2605.25852},
     year    = {2026},
     doi     = {10.48550/arXiv.2605.25852}
   }

.. toctree::
   :hidden:

   getting-started
   methods
   tutorial
   modules
