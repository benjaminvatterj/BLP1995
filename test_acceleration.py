from BLP import BLP
import numpy as np
from test import Faker
from time import time

"""
This file provides some testing
for the acceleration algorithm
"""

if __name__ == "__main__":

    market = Faker()
    market.genData(150, 10, 5, 500)
    blp = BLP(market.X1, market.X2, market.Z, market.M, market.S)
    blp.prepareSample()
    a = np.random.rand(market.X2.shape[1])
    bounds = [(-5, 5) for x in a]

    # Keep initial delta vector
    initial_delta = blp.initial_delta.copy()

    # Test Anderson-accelerated convergence
    np.copyto(blp.initial_delta, initial_delta)
    print("Starting Anderson accelerated...")
    blp.delta_method = 1
    blp.anderson = np.empty((6, blp.initial_delta.shape[0]))
    start3 = time()
    res3 = blp.solve(a, method="Nelder-Mead", delta_method="anderson")
    end3 = time()

    # Test non-accelerated convergence
    np.copyto(blp.initial_delta, initial_delta)
    print("Starting non-accelerated...")
    blp.delta_method = 0
    start4 = time()
    res4 = blp.solve(a, method="Nelder-Mead", delta_method="picard")
    end4 = time()

    # Test Anderson-accelerated convergence
    print("Starting BFGS Anderson accelerated...")
    blp.delta_method = 1
    blp.anderson = np.empty((6, blp.initial_delta.shape[0]))
    start2 = time()
    res2 = blp.solve(a, method="BFGS", delta_method="anderson", bounds=bounds)
    end2 = time()

    # Test non-accelerated convergence
    np.copyto(blp.initial_delta, initial_delta)
    print("Starting BFGS non-accelerated...")
    blp.delta_method = 0
    start1 = time()
    res1 = blp.solve(a, method="BFGS", delta_method="picard", bounds=bounds)
    end1 = time()

    print("BFGS non-accelerated time : %f sec" % (end1 - start1))
    print("BFGS Anderson-accelerated time: %f sec" % (end2 - start2))
    print("BFGS Distance between results: %e" % (np.linalg.norm(res1.theta2 - res2.theta2, ord=2)))
    print("Nelder-Mead non-accelerated time : %f sec" % (end4 - start4))
    print("Nelder-Mead Anderson-accelerated time: %f sec" % (end3 - start3))
    print("Nelder-Mead Distance between results: %e" % (np.linalg.norm(res3.theta2 - res4.theta2, ord=2)))
