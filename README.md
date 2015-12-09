# Berry Levinsohn Pakes 1995 algorithm implementation
This repository provides a class implementation of BLP(1995)
for Python 3.+, based upon several guiding principles written
by Aviv Nevo in his well known Practitioner Guide.

Some of the features of this BLP implementations are:

1. Flexible simulation of population. Modifiable by user-provided distributions.
2. Two algorithms for the optimization routine:
  1. Nelder-Mead non-derivative search. Better suited for an initial exploration, as it is less sensible to starting points but slower.
  2. L-BFGS-B gradient based search. A bounded BFGS algorithm that has better performance and helps avoid the common overflows due to the exponential form of logit.
3. Adaptive delta fixed-point tolerance (as recommended by Aviv Nevo).
4. Flexible separation of the fixed point by markets. This allows for:
  1. Solving the entire data each step.
  2. Separating by groups and solving each sequentially.
  3. Separating by groups and solving in parallel on a multiprocessor computer.
5. Optional Anderson acceleration of the delta fixed-point.
6. A lot of comments to guide anyone who wants to implement his one version.

However, the algorithm assumes no correlation between the individual taste coefficients, that is Sigma is a diagonal as in the Nevo guide, which is relatively standard in the literature.


Additionally to the main BLP.py class, the file test.py provides
a fairly random Monte-Carlos simulator that generate an unbalanced panel
for testing the class. The package also provides three tests:

1. test.py: test the general functioning of the class
2. test_acceleration.py: shows the advantages of the Anderson acceleration
3. test_mul.py: shows the advantages of the multiprocessing technique.

test_log.txt shows the results of the last one, that also showcases the other
points.

So far, the recommended workflow for this class is to first preform a few high-tolerance runs with different starting points using the Nelder-Mead algorithm.
Once you have figured out where the results might be, do a low-tolerance bounded run with the L-BFGS-B algorithm.
In general, Anderson acceleration, adaptive delta tolerance are recommended. Parallelization is recommended only if each fixed-point calculation takes long enough to justify the overhead implied in the parallelization.


See notes.txt for a short description of the algorithm and
further details about possible future updates to this package.

See the test files for examples of how to easily use this class. Note that though it is quite flexible, very few parameters are actually required to be inputed for it to work.


updates:
-------
* Improved memory sharing between parallel processes.
* Improved Jacobian calculation algorithm .
* Improved delta fixed point algorithm (faster population summations).
