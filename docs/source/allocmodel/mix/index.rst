=======================
Mixture Models
=======================

**bnpy** supports two kinds of mixture models: `FiniteMixtureModel` and `DPMixtureModel`. 

`FiniteMixtureModel`
--------------------
The finite mixture has the following generative representation as an allocation model. There is a single top-level vector of cluster probabilities :math:`\pi_0`. Each data atom's assignment is drawn i.i.d. according to the probabilities in this vector.

.. math::

	[\pi_{01}, \pi_{02}, \ldots \pi_{0K}] 
	\sim \mbox{Dir}_K( \frac{\alpha_0}{K} )
	\\
	\mbox{for~} n \in 1, \ldots N:
	\\
	\qquad z_n \sim \mbox{Cat}_K(\pi_{01}, \lddots \pi_{0K})

Here, :math:`\alpha_0 > 0` is the uniform concentration parameter. TODO interpret.

`DPMixtureModel`
--------------------
The Dirichlet Process (DP) mixture has the following generative representation as an allocation model. It modifies the finite mixture by using the StickBreaking process to K active weights and a remainder weight, all inside $\pi_0$.

.. math::
	[\pi_{01}, \pi_{02}, \ldots \pi_{0K}, \pi_{0, >K}] 
	\sim \mbox{StickBreaking}_K(\pi_0)
	\\
	\mbox{for~} n \in 1, \ldots N:
	\\
	\qquad z_n \sim \mbox{Cat}_K(\pi_{01}, \lddots \pi_{0K})

If we take the limit as K grows to infinity, these two generative models are equivalent.

Using mixtures with other **bnpy** modules
------------------------------------------

As usual, to train a hierarchical model whose allocation is done by FiniteMixtureModel,  

.. code::
	>>> hmodel, Info = bnpy.Run(Data, 'FiniteMixtureModel', obsModelName, algName, **kwargs)
	>>> # or
	>>> hmodel, Info = bnpy.Run(Data, 'DPMixtureModel', obsModelName, algName, **kwargs)

Supported DataObj Types
+++++++++++++++++++++++++++

Mixture models can apply to almost all data formats available in bnpy.
Any data suitable for topic models or sequence models can also be fit
with a basic mixture model.

The only formats that do not apply are those based on GraphData, 
which require the subclass of mixture models (TBD).

Supported Learning Algorithms
+++++++++++++++++++++++++++++
Currently, the practical differences are:

* `FiniteMixtureModel` supports EM, VB, soVB, moVB
* `DPMixtureModel` supports VB, soVB, and moVB.
* * with birth/merge/delete moves for moVB

EM (MAP) inference for the DPMixtureModel is possible, but just not implemented yet.



Common tasks with mixtures
---------------------------

Accessing learned cluster assignments
+++++++++++++++++++++++++++++++++++++

Given a dataset of interest Data (a :class:`.DataObj`), and an hmodel (an instance of :class:`.HModel`) properly initialized with K active clusters, we simply perform a local step.

.. code::

	>>> LP = hmodel.calc_local_params(Data)
	>>> resp = LP['resp']

Here, resp is a 2D array of size N x K. 
Each entry resp[n, k] gives the probability that data atom n is assigned to cluster k under the posterior. 
Thus, each entry resp[n,k] must be a value within the interval [0,1].
The sum of every row must equal one.

.. code::

	>>> assert resp[n, k] >= 0.0
	>>> assert resp[n, k] <= 1.0
	>>> assert np.allclose(np.sum(resp[n,:]), 1.0)


To convert to hard assignments

.. code::

	>>> Z = resp.argmax(axis=1)

Here, Z is a 1D array of size N, where entry Z[n] is an integer in the set {0, 1, 2, ... K-1, K}.

Accessing learned cluster probabilities
+++++++++++++++++++++++++++++++++++++++

.. code::

	>>> pi0 = hmodel.allocModel.get_active_cluster_probs()
	>>> assert pi0.ndim == 1
	>>> assert pi0.size == hmodel.allocModel.K

Global update summaries
+++++++++++++++++++++++++++

For a global update, mixture models require only one sufficient statistic: an expected count value for each cluster k. This value gives the expected number of data atoms assigned to k throughout the dataset.

* Count N_k
	Expected assignments to state k across all data items.

.. code::

	>>> LP = hmodel.calc_local_params(Data)
	>>> SS = hmodel.get_global_suff_stats(Data, LP)
	>>> Nvec = SS.N # or SS.getCountVec()
	>>> assert Nvec.size == hmodel.allocModel.K
	[ ... TODO ... ]

ELBO summaries
++++++++++++++

To compute the ELBO, mixture models require only one non-linear summary statistic: the entropy of the learned assignment parameters `resp`.

.. math::

	\L = \Ldata + \Lalloc - E[ \log q(z) ]

	- E[ \log q(z) ] = \sum_{k=1}^K H_k

	H_k = - \sum_{n=1}^N r_{nk} \log r_{nk}

You can compute this by enabling the correct keyword flag when calling the summary step function.

.. code::

	>>> LP = hmodel.calc_local_params(Data)
	>>> SS = hmodel.get_global_suff_stats(Data, LP, doPrecompEntropy=1)
	>>> Hresp =  SS.getELBOTerm('Hresp')
	>>> assert Hresp.ndim == 1
	>>> assert Hresp.size == SS.K
	[ ... TODO ... ]

.. toctree::
   :maxdepth: 3
   :titlesonly:
   :hidden:

   FiniteMixtureModel.rst
   FiniteMixtureModel-Variational.rst

   DPMixtureModel.rst
   DPMixtureModel-Variational.rst