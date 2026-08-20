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

References
----------

* **Split conformal prediction:** Jing Lei, Max G’Sell, Alessandro Rinaldo, Ryan J.
  Tibshirani, and Larry Wasserman. “Distribution-Free Predictive Inference for
  Regression.” *Journal of the American Statistical Association*, 113(523), 1094–1111,
  2018. `DOI <https://doi.org/10.1080/01621459.2017.1307116>`_.
* **Conformalized quantile regression:** Yaniv Romano, Evan Patterson, and Emmanuel J.
  Candès. “Conformalized Quantile Regression.” *Advances in Neural Information
  Processing Systems 32*, 2019.
  `NeurIPS <https://papers.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html>`_.
* **HPD-split:** Rafael Izbicki, Gilson Shimizu, and Rafael B. Stern. “CD-split and
  HPD-split: Efficient Conformal Regions in High Dimensions.” *Journal of Machine
  Learning Research*, 23(87), 1–32, 2022.
  `JMLR <https://www.jmlr.org/papers/v23/20-797.html>`_.
* **CONTRA:** Zhenhan Fang, Aixin Tan, and Jian Huang. “CONTRA: Conformal Prediction
  Region via Normalizing Flow Transformation.” *International Conference on Learning
  Representations*, 2025.
  `ICLR <https://proceedings.iclr.cc/paper_files/paper/2025/hash/e55d081280e79e714debf2902e18eb69-Abstract-Conference.html>`_.
