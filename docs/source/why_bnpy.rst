==========
Why bnpy?
==========

Welcome! **bnpy** is a Python package for performing Bayesian nonparametric unsupervised learning. 
This document explains the key motivating ideas behind **bnpy**. 
For readers new to this field of machine learning, we hope it gives you a high-level introduction and has many links to accessible references with more details.


Motivation: What's wrong with kmeans?
-------------------------------------

Our core applications at **bnpy** is unsupervised clustering. 
In simplest form, the task is to take a collection of datapoints and assign each one to a specific cluster (also called topic, state, etc.).

Any student who has taken a "machine learning 101" class will naturally suggest the k-means algorithm to solve this task, or perhaps its more sophisticated cousin the EM algorithm. Both of these treat the task as an optimization, and can be interpreted as "maximum likelihood" (or "minimum energy"). 

Maximum likelihood methods are often a fine first step, but there are three crucial problems with this approach:

* Fail to represent uncertainty
* Get stuck in local optima
* Lack any penalty on the number of clusters


Bayesian nonparametric models
-----------------------------


Scalable inference that escapes local optima
--------------------------------------------
