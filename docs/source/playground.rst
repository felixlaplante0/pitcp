Interactive playground
======================

Explore the same heteroscedastic candy example used by the experiment scripts. Choose
``PITCP``, ``SCP``, ``CQR``, ``HPD``, or ``CONTRA`` and request any confidence level
from 0.50 to 0.99.

The density-based models load fitted, confidence-independent artifacts. ``CQR`` is
fitted for the confidence level you select because its response quantiles change with
that level. Results use 2,500 test observations, all of which are shown in the
region plot.

.. raw:: html

   <iframe
     src="https://pitcp.streamlit.app/?embed=true"
     title="PIT-CP interactive playground"
     loading="lazy"
     allow="clipboard-read; clipboard-write"
     style="width: 100%; height: 1050px; border: 1px solid var(--pst-color-border); border-radius: 12px;"
   ></iframe>

If the embedded application is sleeping or does not fit your screen,
`open the playground in a new tab <https://pitcp.streamlit.app/>`_.
