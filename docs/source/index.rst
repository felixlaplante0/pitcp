PIT-CP
======

**pitcp** is a Python package for conformal prediction using probability integral transform (PIT) pivotal scores. Given any black-box nonconformity score, it fits a conditional density estimator on the score distribution and maps raw scores to PIT values, yielding valid marginal coverage at any user-specified level.

Our contribution is `PITCP`. The package also reimplements the state-of-the-art `SCP`, `CQR`, `HPD`, and `CONTRA` baselines behind a consistent scikit-learn API.

Features
--------

- **PIT conformal prediction**: `PITCP` maps base nonconformity scores through a learned conditional CDF.
- **Split conformal prediction**: `SCP` calibrates arbitrary scalar nonconformity scores without a learned correction.
- **Conformalized quantile regression**: `CQR` accepts multiple outputs and provides a scikit-learn gradient-boosting implementation of state-of-the-art conformalized quantile regression.
- **Highest-density regions**: `HPD` calibrates conditional highest-predictive-density sets.
- **Latent-space regions**: `CONTRA` maps targets through a conditional normalizing flow and calibrates an Euclidean norm-based score in latent space.
- **Conformal utilities**: Computes coverage gaps and region volumes for every supported region type.

Installation
------------

Install the package from PyPI:

.. code-block:: bash

   python -m pip install pitcp

Usage
-----

The following example fits a conditional score distribution, calibrates it on held-out data, and predicts score thresholds and coverage indicators.

.. code-block:: python

   import torch
   import zuko
   from pitcp import PITCP


   def std(x):
       return torch.where((x > -0.9) & (x < 0.9), torch.cos(torch.pi * x / 2), 1.0)


   def gen_data(n):
       x = torch.rand(n, 1) * 2 - 1
       return x, torch.randn(n, 1) * std(x)


   torch.manual_seed(42)
   (X_train, y_train), (X_cal, y_cal), (X_test, y_test) = [
       gen_data(5000) for _ in range(3)
   ]

   s_train = y_train.abs()
   s_cal = y_cal.abs()
   s_test = y_test.abs()
   density = zuko.flows.NSF(
       features=1, context=1, bins=4, hidden_features=(32, 32, 32)
   )
   optimizer = torch.optim.Adam(density.parameters(), lr=1e-2)

   model = PITCP(density, optimizer, n_epochs=10, batch_size=128)
   model.fit(X_train, s_train).conformalize(X_cal, s_cal)

   limits = model.predict(X_test, confidence_level=[0.7, 0.8, 0.9])
   covered = model.contains(
       X_test, s_test, confidence_level=[0.7, 0.8, 0.9]
   )

Configuration
-------------

`PITCP` learns conditional score quantiles through a normalizing flow or Gaussian mixture. `SCP` calibrates scores directly. `CQR` learns lower and upper conditional quantiles. `HPD` calibrates density ranks, and `CONTRA` calibrates an Euclidean norm-based score in latent space.

All estimators follow the scikit-learn parameter API. Fitted estimators expose calibration scores through scores\_. Density-based estimators accept n_epochs, batch_size, verbose, and random_state training controls.

API Reference
-------------

.. toctree::
   :maxdepth: 2

   modules
