Getting started
===============

Install ``pitcp`` from PyPI:

.. code-block:: bash

   python -m pip install pitcp

The conformal workflow uses separate training and calibration data. Training learns
the conditional score distribution; calibration supplies the finite-sample coverage
guarantee.

.. code-block:: python

   import torch
   import zuko
   from pitcp import PITCP

   density = zuko.flows.SOSPF(
       features=1, context=1, hidden_features=(32, 32)
   )
   optimizer = torch.optim.Adam(density.parameters(), lr=1e-3)
   model = PITCP(
       density,
       optimizer,
       n_epochs=200,
       batch_size=512,
       random_state=42,
   )

   model.fit(X_train, scores_train)
   model.conformalize(X_cal, scores_cal)
   limits = model.predict(X_test, confidence_level=0.9)
   covered = model.contains(X_test, scores_test, confidence_level=0.9)

What each split does
--------------------

.. grid:: 1 3 3 3

   .. grid-item-card:: 1. Train

      Learn the conditional density of the chosen score.

   .. grid-item-card:: 2. Calibrate

      Store held-out pivotal scores for conformal quantiles.

   .. grid-item-card:: 3. Predict

      Request one confidence level or several from the fitted model.

Next, work through the :doc:`tutorial` or compare methods in the
:doc:`playground`.
