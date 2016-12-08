======================
Allocation Model Guide
======================

What is an allocation model?
----------------------------

Within **bnpy**, every hierarchical model we support has two pieces: an allocation model and an observation model.
We use the label "allocation model" to describe the generative process that allocates cluster assignments to individual data points.

TODO ILLUSTRATION

In this document, we give a high-level overview of how we define an allocation model and how variational inference works. We also define the essential variational inference API functions that any concrete allocation model (an instance of the abstract :class:`AllocModel` class) should support.

Quick Links
-----------
Here are some quick links to documentation for each of the possible allocation models supported by bnpy.

* :doc:`Mixture models <mix/index>`

* :doc:`Topic models <topics/index>`

* :doc:`Hidden Markov models <hmm/index>`


Generative model
----------------
An allocation model defines a probabilistic generative process for assigning (aka allocating) clusters to data atoms. There are two types of variables involved: cluster probability vectors $\pi_j$, and discrete assignments $z_n$ at each data aton indexed by $n$.  Each allocation model defines a joint distribution

.. math::
	\log p(\pi, z) = \log p(\pi) + \log p( z | \pi)

First, we generate a set of global cluster probabilities $\pi_0$. 

.. math::
	\pi_0 \sim \mbox{Dir}_K(\frac{\alpha_0}{K})

Depending on the model, we may next generate several more cluster probability vectors $\pi_j$. 

Second, we draw cluster assignment variables $z_n$ at each data atom $n$.

.. math::
	z_n \sim \mbox{Cat}( \pi_{j1}, \ldots \pi_{jK} )


Example: Mixture model
++++++++++++++++++++++
For example, consider a simple finite mixture model with $K$ clusters. The complete allocation model would be:

.. math::
	\pi_0 \sim \mbox{Dir}_K(\alpha_0 \frac{1}{K})

	z_n \sim \mbox{Cat}( \pi_{01}, \ldots \pi_{0K} )

To extend this to a Dirichlet process mixture model, we simply use a stick-breaking distribution instead:

.. math::
	\pi_0 \sim \mbox{Stick}(\alpha_0)

	z_n \sim \mbox{Cat}( \pi_{01}, \ldots \pi_{0K}, \ldots)


Variational Inference
---------------------

Variational inference for allocation models tries to optimize an approximate posterior:

.. math::
	\log q(\pi, z) = \log q(\pi | \theta) + \log q(z | r)

The optimization objective is to make this approximate posterior as close to the true posterior as possible. Remember that this objective incorporates terms from the observation model as well. The optimization finds values for the free parameters -- pseudo-counts \theta and assignments r -- that make the objective function as large as possible.

.. math::
	\L = \Lalloc(r, theta) + \Lobs(r, ...)

Expanding the allocation model terms, we have

.. math::
	\Lalloc(r, theta) = \Lz + \Lentropy

	\Lz = \E_q[ \log p(z) + \frac{\log p(\pi)}{\log q(\pi)} ]

	\Lentropy = - \E_q[ \log q(z) ]

	\log p(z | \alpha) \geq \Lpz

Every variational algorithm proceeds by iteratively improving this objective function by cycling through four concrete steps:

* Local step: optimize the local assignments r and any local theta values.
* Summary step: compute summary statistics from the local parameters.
* Global step: 
* Objective function evaluation step


Variational API
---------------
Within **bnpy**, each possible allocation model is a subclass of the general-purpose abstract base class: :class:`AllocModel`. 
Each :class:`AllocModel` instance has both state and behaviors.
The *state* represents two key values: the hyperparameters that define the prior and the global variational parameters that define the approximate posterior. The *behaviors* are the four fundamental steps of inference, as well as some auxiliary functions.


Attributes
++++++++++
For any generative model in our framework, the hyperparameters of an allocation model are just the set of concentration parameters $\alpha_j$ that parameterize the generative story for each $\pi_j$ probability vector.
Thus, each allocation model will hold one or more `alpha` values as attributes.

Each :class:`AllocModel` subclass will have model-specific global parameters, which are represented as instance attributes. For example, a :class:`FiniteMixtureModel` has a vector of Dirichlet pseudo-counts called `theta`, while a :class:`DPMixtureModel` instance has a vector of Beta pseudo-counts called `eta`.  


Each of the four conceptual steps of the variational inference -- local step, summary step, global step, and objective step -- is associated with a single instance-level function of an AllocModel object. The general abstract interface for using these functions is documented below. Each subclass will provide an actual implementation of these functions.

Local step
++++++++++

The local step, specified by calc_local_params, finds local parameters for the dataset.

.. currentmodule:: bnpy

.. autoclass:: bnpy.allocmodel.AllocModel
   :members: calc_local_params



Summary step
++++++++++++ 

The summary step, specified by get_global_suff_stats, summarizes a dataset Data and its associated local parameters LP. It produces a bag of sufficient statistics SS.

.. currentmodule:: bnpy

.. autoclass:: bnpy.allocmodel.AllocModel
   :members: get_global_suff_stats


Global step
+++++++++++

The global step, performed by update_global_params, 

.. currentmodule:: bnpy

.. autoclass:: bnpy.allocmodel.AllocModel
   :members: get_global_suff_stats


Objective evaluation step
+++++++++++++++++++++++++

During inference, we need to verify that each step is working as expected. Thus, we need to be able to compute the scalar value of the objective given any current set of global parameters (stored in self) and local parameters (summarized in SS).

.. currentmodule:: bnpy

.. autoclass:: bnpy.allocmodel.AllocModel
   :members: calc_evidence


.. toctree::
   :maxdepth: 3
   :hidden:
   :titlesonly:

   mix/index
   topics/index
   hmm/index
