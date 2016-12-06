============
Installation
============

Prerequisites
=============
**bnpy** depends on Python and the following (external) Python packages

* numpy, version > 1.8
* scipy, version > 1.10

Optionally, we also require these packages for visualization:

* matplotlib


How to install prerequisites (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We highly recommend the Enthought Python distribution. 

It's one-click install comes with all the requirements for **bnpy** and avoids the hassle of individually installing each package.

Furthermore, the numerical subroutines for matrix operations that ship with the EPD are almost always better than what a novice user can build themselves from source or have installed from other routes. We've routinely observed speedups of 2-4x on basic operations like matrix multiplication.

How to install prerequisites (advanced)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can either build numpy and scipy from source, or install via a package manager like "easy_install" or "pip". If you must go this route, we recommend using pip.  You can search the web for the latest and greatest instructions for installing these.

Installing bnpy
=================

You can grab the latest stable version from our master git repository.  

Execute the following command to have the project file structure cloned to your local disk, within a folder called "bnpy"

.. code-block:: bash

	git clone https://michaelchughes@bitbucket.org/michaelchughes/bnpy-dev.git


Throughout all documentation, we'll call the directory where **bnpy** is installed `$BNPYROOT`.  Wherever you see this, substitute in the actual directory on your system.

If you execute `ls $BNPYROOT`, you should see `bnpy/`, `demodata/`, and several other files and folders.  


Verifying installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open a terminal, and type `python` to drop into a python shell.  Then type:

.. code-block:: python

	import numpy
	print numpy.__version__

If this works OK, you've got numpy working. To further verify matplotlib installation, enter:

.. code-block:: python

	from matplotlib import pylab
	pylab.plot([1,2,3])
	pylab.show()

If that produces a figure with a simple line plot, you're good to go!

.. code-block:: python

	import bnpy

Common errors with matplotlib
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


If you try the above and get errors about not having "wx" or "wxpython" or "qt" installed, you need to configure your Matplotlib_backend_.

.. _Matplotlib_backend: http://matplotlib.org/faq/usage_faq.html#what-is-a-backend


I recommend setting your matplotlibrc file to have `backend: TkAgg` for Linux, and `backend: MacOSX` for Mac.



Configuration
==============

After you have installed a copy of **bnpy** on your system, you need to adjust a few key settings to make sure that you are ready to use **bnpy**.  Here, we introduce the concept of *environment variables*, and discuss how **bnpy** uses them to accomplish three key tasks.

* Tell Python where to find the "bnpy" module
* Tell bnpy where to save results
* (optional) Tell bnpy where to find your custom datasets.

**What is an environment variable?** 
In practice, environment variables allow you (the user) to define locations on your system (where to read data, where to save results, etc.), without these needing to be hard-coded into the **bnpy** module or passed as an argument everytime **bnpy** runs.

**Simple Example:**  Open a terminal and try this. 

.. code-block:: bash

	$ MYVAR=42
	$ echo $MYVAR
	42

You've just set an environment variable to 42 and then printed its value.

In most UNIX systems, the keyword `export` makes a variable global, so that other processes (like python) can read that variable's value.

.. code-block:: bash

	export MYVAR=42

Setting up bnpy environment variables.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1:** Tell Python where to find the **bnpy** module

Python always looks at the environment variable called PYTHONPATH to find custom-installed modules.

.. code-block:: bash

	export PYTHONPATH=/path/to/bnpy/


**Step 2:** Tell **bnpy** where to save results

**bnpy** looks at `BNPYOUTDIR` to define the complete path of the directory where results are saved.

.. code-block:: bash

	export BNPYOUTDIR=/path/to/my/results/

Make sure this directory is readable and writeable by you.  Also make sure it has enough free disk space (a few GBs will do just fine) if you plan to do extensive experimentation.  

**Step 3 (optional):** Tell **bnpy** where to load custom datasets from

By default, **bnpy** will already know how to find the pre-installed toy and real datasets. However, to run **bnpy** on custom, user-defined data, you will need to specify a location. 

**bnpy** can process any dataset defined in a dataset script. The location of these scripts are specified by the Unix environment variable *BNPYDATADIR*.

.. code-block:: bash

	export BNPYDATADIR=/path/to/my/custom/dataset/

In general, you might change this location every time you work with a different custom dataset.


**If you are using IDE:**  If you choose to develop and run your code in IDE, then the configurations need to be set somewhere else. Here we take PyCharm 3.4 on Mac OS X as an example. In the menu bar of PyCharm, select `Run -> Edit Configurations...`. Then in `Environment -> Environment Variables`, manually add the environment variables mentioned above as key value pairs (e.g. PYTHONPATH /path/to/bnpy/) and press OK. 

A more general (but dangerous) way to do this is that you can edit the file `/etc/launchd.conf` in your machine to add these variables by writing down commands like

.. code-block :: bash

	setenv PYTHONPATH /path/to/bnpy

into the file and restart your machine. In Pycharm, the second method could keep the autocompletion working when you deal with stuff in **bnpy** module.
