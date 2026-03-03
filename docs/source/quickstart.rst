Quickstart
==========

Convert any PyTorch model into a Bayesian Neural Network:

.. code-block:: python

   import torchbayesian as tb

   model = MyModel()
   bnn_model = tb.BayesianModule(model)
