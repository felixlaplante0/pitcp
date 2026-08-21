PIT-CP
======

.. raw:: html

   <section class="hero">
     <img class="hero-logo" src="_static/pitcp-logo.svg" alt="PIT-CP adaptive prediction logo">
     <p class="eyebrow">CONFORMAL PREDICTION, MADE ADAPTIVE</p>
     <h1>Prediction regions that follow the data.</h1>
     <p class="hero-copy">PIT-CP learns how a nonconformity score changes with the
     input, then conformalizes it at any confidence level.</p>
     <div class="hero-actions">
       <a class="primary" href="getting-started.html">Get started</a>
       <a class="secondary" href="https://pitcp-app.streamlit.app/">Try the playground</a>
     </div>
   </section>

.. raw:: html

   <aside class="pypi-card">
     <div>
       <span class="pypi-kicker">PYTHON PACKAGE</span>
       <strong>Available on PyPI</strong>
       <p>Install PIT-CP in one command and start building adaptive regions.</p>
     </div>
     <a href="https://pypi.org/project/pitcp/">View package&nbsp;→</a>
   </aside>

Why PIT-CP?
-----------

Marginal coverage can hide large local errors. ``PITCP`` learns the conditional
distribution of any scalar nonconformity score and maps it to a pivotal score before
calibration. The same fitted density supports every confidence level.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Adaptive regions
      :class-card: feature-card

      Let prediction widths follow heteroscedasticity instead of staying constant.

   .. grid-item-card:: Any confidence level
      :class-card: feature-card

      Change the requested coverage without fitting the PITCP density again.

   .. grid-item-card:: Familiar workflow
      :class-card: feature-card

      Fit, conformalize, predict, and evaluate with scikit-learn-style estimators.

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

.. toctree::
   :hidden:
   :maxdepth: 2

   getting-started
   methods
   tutorial
   modules
