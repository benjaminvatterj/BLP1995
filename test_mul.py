from BLP import BLP
import numpy as np
from test import Faker
from time import time

"""

This file tests the different
options of parallelization
of the delta fixed points

"""


if __name__ == "__main__":

    market = Faker()
    market.genData(150, 10, 3, 500)
    a = np.random.rand(market.X2.shape[1])
    bounds = [(-5, 5) for x in a]

    # Test serial unique convergence
    blp = BLP(market.X1, market.X2, market.Z, market.M, market.S)
    blp.prepareSample()
    population = blp.population
    population_size = blp.population_size  # Keep these for easier comparison
    print("Starting picard serial...")
    start1 = time()
    res1 = blp.solve(a, method="Nelder-Mead", delta_method="picard")
    end1 = time()
    blp = None

    # Test serial multiple convergence
    blp2 = BLP(market.X1, market.X2, market.Z, market.M, market.S, par_cut=2)
    blp2.population = population
    blp2.status = 1
    blp2.population_size = population_size
    print("Starting picard split...")
    start2 = time()
    res2 = blp2.solve(a, method="Nelder-Mead", delta_method="picard")
    end2 = time()
    blp2 = None

    # Test parallel convergence
    blp3 = BLP(market.X1, market.X2, market.Z, market.M, market.S, parallel=True, threads=2)
    blp3.population = population
    blp3.status = 1
    blp3.population_size = population_size
    print("Starting picard parallel...")
    start3 = time()
    res3 = blp3.solve(a, method="Nelder-Mead", delta_method="picard")
    end3 = time()
    blp3 = None

    # Test serial unique convergence
    blp = BLP(market.X1, market.X2, market.Z, market.M, market.S)
    blp.population = population
    blp.status = 1
    blp.population_size = population_size
    print("Starting Anderson accelerated serial...")
    start4 = time()
    res4 = blp.solve(a, method="Nelder-Mead", delta_method="anderson")
    end4 = time()
    blp = None

    # Test serial multiple convergence
    blp2 = BLP(market.X1, market.X2, market.Z, market.M, market.S, par_cut=2)
    blp2.population = population
    blp2.status = 1
    blp2.population_size = population_size
    print("Starting Anderson accelerated split...")
    start5 = time()
    res5 = blp2.solve(a, method="Nelder-Mead", delta_method="anderson")
    end5 = time()
    blp2 = None

    # Test parallel convergence
    blp3 = BLP(market.X1, market.X2, market.Z, market.M, market.S, parallel=True, threads=2)
    blp3.population = population
    blp3.status = 1
    blp3.population_size = population_size
    print("Starting Anderson accelerated parallel...")
    start6 = time()
    res6 = blp3.solve(a, method="Nelder-Mead", delta_method="anderson")
    end6 = time()
    blp3 = None

    # Test serial unique convergence
    blp = BLP(market.X1, market.X2, market.Z, market.M, market.S)
    blp.population = population
    blp.status = 1
    blp.population_size = population_size
    print("Starting picard serial...")
    start7 = time()
    res7 = blp.solve(a, method="BFGS", delta_method="picard", bounds=bounds)
    end7 = time()
    blp = None

    # Test serial multiple convergence
    blp2 = BLP(market.X1, market.X2, market.Z, market.M, market.S, par_cut=2)
    blp2.population = population
    blp2.status = 1
    blp2.population_size = population_size
    print("Starting picard split...")
    start8 = time()
    res8 = blp2.solve(a, method="BFGS", delta_method="picard", bounds=bounds)
    end8 = time()
    blp2 = None

    # Test parallel convergence
    blp3 = BLP(market.X1, market.X2, market.Z, market.M, market.S, parallel=True, threads=2)
    blp3.population = population
    blp3.status = 1
    blp3.population_size = population_size
    print("Starting picard parallel...")
    start9 = time()
    res9 = blp3.solve(a, method="BFGS", delta_method="picard", bounds=bounds)
    end9 = time()
    blp3 = None

    # Test serial unique convergence
    blp = BLP(market.X1, market.X2, market.Z, market.M, market.S)
    blp.population = population
    blp.status = 1
    blp.population_size = population_size
    print("Starting Anderson accelerated serial...")
    start10 = time()
    res10 = blp.solve(a, method="BFGS", delta_method="anderson", bounds=bounds)
    end10 = time()
    blp = None

    # Test serial multiple convergence
    blp2 = BLP(market.X1, market.X2, market.Z, market.M, market.S, par_cut=2)
    blp2.population = population
    blp2.status = 1
    blp2.population_size = population_size
    print("Starting Anderson accelerated split...")
    start11 = time()
    res11 = blp2.solve(a, method="BFGS", delta_method="anderson", bounds=bounds)
    end11 = time()
    blp2 = None

    # Test parallel convergence
    blp3 = BLP(market.X1, market.X2, market.Z, market.M, market.S, parallel=True, threads=2)
    blp3.population = population
    blp3.status = 1
    blp3.population_size = population_size
    print("Starting Anderson accelerated parallel...")
    start12 = time()
    res12 = blp3.solve(a, method="BFGS", delta_method="anderson", bounds=bounds)
    end12 = time()
    blp3 = None

    print("1) NM - serial unique time : %f sec" % (end1 - start1))
    print("2) NM - serial split time: %f sec" % (end2 - start2))
    print("3) NM - parallel time: %f sec" % (end3 - start3))
    print("4) NM - Accelerated serial unique time : %f sec" % (end4 - start4))
    print("5) NM - Accelerated serial split time: %f sec" % (end5 - start5))
    print("6) NM - Accelerated parallel time: %f sec" % (end6 - start6))
    print("7) BFGS - serial unique time : %f sec" % (end7 - start7))
    print("8) BFGS - serial split time: %f sec" % (end8 - start8))
    print("9) BFGS - parallel time: %f sec" % (end9 - start9))
    print("10) BFGS - Accelerated serial unique time : %f sec" % (end10 - start10))
    print("11) BFGS - Accelerated serial split time: %f sec" % (end11 - start11))
    print("12) BFGS - Accelerated parallel time: %f sec" % (end12 - start12))

    print("theta2:")
    print("1) %s" % res1.theta2)
    print("2) %s" % res2.theta2)
    print("3) %s" % res3.theta2)
    print("4) %s" % res4.theta2)
    print("5) %s" % res5.theta2)
    print("6) %s" % res6.theta2)
    print("7) %s" % res7.theta2)
    print("8) %s" % res8.theta2)
    print("9) %s" % res9.theta2)
    print("10) %s" % res10.theta2)
    print("11) %s" % res11.theta2)
    print("12) %s" % res12.theta2)

    print("theta1:")
    print("1) %s" % res1.theta1)
    print("2) %s" % res2.theta1)
    print("3) %s" % res3.theta1)
    print("4) %s" % res4.theta1)
    print("5) %s" % res5.theta1)
    print("6) %s" % res6.theta1)
    print("7) %s" % res7.theta1)
    print("8) %s" % res8.theta1)
    print("9) %s" % res9.theta1)
    print("10) %s" % res10.theta1)
    print("11) %s" % res11.theta1)
    print("12) %s" % res12.theta1)
