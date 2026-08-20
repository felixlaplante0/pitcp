Interactive playground
======================

Explore the same heteroscedastic candy example used by the experiment scripts in the
`interactive Streamlit playground <https://pitcp-app.streamlit.app/>`_. Choose
``PITCP``, ``SCP``, ``CQR``, ``HPD``, or ``CONTRA`` and request any confidence level
from 0.50 to 0.99.

The density-based models load fitted, confidence-independent artifacts. ``CQR`` is
fitted for the confidence level you select because its response quantiles change with
that level. Results use 2,500 test observations, all of which are shown in the
region plot.

.. raw:: html

   <meta http-equiv="refresh" content="0; url=https://pitcp-app.streamlit.app/">

`Open the playground <https://pitcp-app.streamlit.app/>`_.
