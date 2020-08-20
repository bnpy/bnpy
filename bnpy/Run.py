'''
User-facing executable script for running experiments that train BNPy models
using a variety of possible inference algorithms, including:

Quickstart
----------

# From the terminal

To run EM for a 3-component GMM on easy, predefined toy data, do

$ python -m bnpy.Run AsteriskK8/x_dataset.csv MixModel Gauss EM --K=3

Can pass *any* kwargs using the --kwarg_name value syntax

'''

import bnpy.Runner

if __name__ == '__main__':
    bnpy.Runner.run()
