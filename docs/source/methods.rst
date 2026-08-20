Method guide
============

All five estimators use held-out calibration data, but they learn different objects.

.. list-table::
   :header-rows: 1
   :widths: 15 35 25 25

   * - Method
     - What it learns
     - Confidence changes
     - Useful when
   * - ``PITCP``
     - Conditional distribution of a scalar score
     - No refit
     - A black-box score needs adaptive thresholds
   * - ``SCP``
     - No predictive model; only calibrated score quantiles
     - No refit
     - A simple marginal-coverage baseline is enough
   * - ``CQR``
     - Lower and upper response quantiles
     - Refit the quantile models
     - Response intervals are the natural output
   * - ``HPD``
     - Conditional response density and density ranks
     - No density refit
     - High-density regions may be preferable to equal tails
   * - ``CONTRA``
     - Conditional flow and latent radii
     - No density refit
     - Invertible multivariate regions are needed

PITCP in one equation
---------------------

For a score :math:`S=s(X,Y)`, PITCP learns its conditional CDF
:math:`\widehat F_{S\mid X}` and forms the pivotal score

.. math::

   \widehat{S} = \widehat F_{S\mid X}(S \mid X).

A split-conformal quantile of held-out values of :math:`\widehat{S}` provides the calibrated
threshold. Prediction inverts the learned conditional CDF, allowing the score limit
to vary with :math:`X`.

The :doc:`tutorial` keeps one dataset and one target confidence level so the
differences between methods stay visible.

This pivotal-score construction is developed in the accompanying paper
[Laplante2026]_.

.. [Laplante2026] Félix Laplante, *A Post-Processing Conformal Prediction Approach
   for Conditional Coverage via Pivotal Scores*, 2026.
   `arXiv:2605.25852 <https://arxiv.org/abs/2605.25852>`_.
