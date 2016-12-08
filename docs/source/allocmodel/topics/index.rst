=======================
Topic Models
=======================

Supported Data Formats
~~~~~~~~~~~~~~~~~~~~~~~

Topic models can be applied to any dataset that has group structure.

Supported Learning Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* `FiniteTopicModel` supports VB, soVB, moVB
* `HDPTopicModel` supports VB, soVB, and moVB.
  * with birth/merge/delete moves for moVB


Possible Implementations
~~~~~~~~~~~~~~~~~~~~~~~~

* FiniteTopicModel: 
  stuff here

* HDPTopicModel:
  more stuff here

There are two types of mixture model supported. Both define the model in 
terms of a global parameter vector :math:`\beta`, where :math:`\beta_k` gives the probability of topic k, and local assignments :math:`z`, where :math:`z_n` indicates which state {1, 2, 3, ... K} is assigned to data item n.

The `FiniteMixtureModel` has a generative process:

.. math::
	[\beta_1, \beta_2, \ldots \beta_K] 
	\sim \mbox{Dir}(\gamma, \gamma, \ldots \gamma)
	\\
	z_n \sim \mbox{Discrete}(\beta)

while the `DPMixtureModel` has generative process:

.. math::
	[\beta_1, \beta_2, \ldots \beta_K \ldots] 
	\sim \mbox{StickBreaking}(\gamma_0)
	\\
	z_n \sim \mbox{Discrete}(\beta)

If we let K grow to infinity, these two models converge if :math:`\gamma = \gamma_0 /K`.




TOC
~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 3
   :titlesonly:

   FiniteTopicModel.rst
   HDPTopicModel.rst
