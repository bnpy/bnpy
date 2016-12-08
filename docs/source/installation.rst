============
Installation
============

Requirements
============
**bnpy** requires Python 2.7+ and the following packages:

* numpy >= 1.11
* scipy >= 0.18
* pandas >= 0.18
* Cython >= 0.25
* joblib >= 0.10
* memory_profiler >= 0.41
* munkres >= 1.0
* numexpr >= 2.6
* psutil >= 5.0
* scikit_learn >= 0.18

For interactivity and visualization, we also recommend:

* ipython >= 5.1
* matplotlib >= 1.5


Easy installation of bnpy
=========================

First, make sure you have a working local install of the Anaconda python distribution, which makes managing common Python packages within userspace a breeze.

.. _anaconda: https://docs.continuum.io/anaconda/install

Then, you can just clone the latest stable version of bnpy via:

.. code-block:: bash

	git clone https://github.com/bnpy/bnpy.git

And then install from the cloned source via:

.. code-block:: bash

	cd bnpy/
	pip install -e .

Verifying correct installation
------------------------------

Within a terminal, you can first verify basic installation with:

.. code-block:: bash

	python -m bnpy.Run --help

You can further train a very simple model:

.. code-block:: bash

	python -m bnpy.Run \
		DATASET_PATH/faithful/faithful.csv \
		FiniteMixtureModel Gauss VB --nLap 1 --K 3


To further verify matplotlib installation, enter:

.. code-block:: python

	from matplotlib import pylab
	pylab.plot([1,2,3])
	pylab.show()


Advanced Installation
=====================

Some of bnpy's advanced features require compiling custom C++ source code for fast algorithms. These aren't needed for basic usage, but do come in handy.

Installing with Eigen C++ libraries
-----------------------------------

The Eigen C++ Matrix template library (>=3.0) is used for:

* fast local step updates for hidden Markov models
* fast local step updates for L-sparse mixtures

If you want these features, go download and install Eigen_ from
`http://www.eigen.tuxfamily.org 
<http://www.eigen.tuxfamily.org>`_.	

.. _eigen: http://eigen.tuxfamily.org/

To install bnpy with Eigen support, you need to set the following environment variable:

.. code-block:: bash

	export EIGENPATH=/path/to/eigen/

You can verify the right location by verifying the following directory exists:

.. code-block:: bash

	ls $EIGENPATH/Eigen/

If the $EIGENPATH env variable is set when you perform **pip install**, the required C++ libraries should be built and useful automatically.


Installing with Boost C++ math libraries
----------------------------------------

The Boost C++ math library (>= 1.52) is used for the following features:

* fast local step updates for L-sparse topic models

If you want these features, go download and install boost_ from
`http://www.boost.org 
<http://www.boost.org>`_.	

.. _boost: http://www.boost.org/

To install bnpy with Boost C++ support, you need to set the following environment variable:

.. code-block:: bash

	export BOOSTMATHPATH=/path/to/boost/include/

You can verify the right location by verifying the following directory exists:

.. code-block:: bash

	ls $BOOSTMATHPATH/math/

If the $BOOSTMATHPATH env variable is set when you perform **pip install**, the required C++ libraries should be built and useful automatically.


Common errors with matplotlib
=============================

If you try the above and get errors about not having "wx" or "wxpython" or "qt" installed, you need to configure your Matplotlib_backend_.

.. _Matplotlib_backend: http://matplotlib.org/faq/usage_faq.html#what-is-a-backend


I recommend setting your matplotlibrc file to have `backend: TkAgg` for Linux, and `backend: MacOSX` for Mac.

