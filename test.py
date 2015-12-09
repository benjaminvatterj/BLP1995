import numpy as np
from BLP import BLP
from math import ceil

"""
This file test the BLP.py class by
generating a Monte-Carlo simulation.
The simulation data is generated fairly
randomly, which might sometimes
lead to datasets that are difficult
to work with. Nevertheless, if run
a couple of times, it is likely to observe
the precision of the implemented algorithm.


Note:
----
Don't expect the test to work every
time. There is a lot fine-tuning to
do in each run and it is possible that
this automatic run will fail due to this
lack of tuning. The system will provide
more information on errors if they occur.
"""


class Faker:

    """
    A data simulator class that
    also test the BLP class

    Attributes:
    -----------
    S : ndarray
        shares
    M : ndarray
        market index
    X1 : ndarray
       the linear parameters (the first is price)
    X2 : ndarray
        the nonlinear parameters
    Z : ndarray
        the instruments
    alpha : float
        price coefficient
    beta : ndarray
        linear characteristics coefficients
    sigma : ndarray
        nonlinear characteristics coefficients
    """

    def __init__(self):
        return

    def genData(self, markets, products, characteristics, population):
        """
        Generates a data set according with the BLP
        model.

        arguments:
        ----------
        markets : int
            number of markets to simulate
        products : int
            max number of products in a market.
            The panel will be unbalanced.
        characteristics : int
            the number of characteristics
            per products.
        population : int
            the number of individual to
            simulate in each market.

        Returns:
        --------

        """
        # -- Taste parameters -- #
        nonlinear_chars = max([ceil(0.2*characteristics), ceil(np.random.rand()*characteristics)])
        nonlinear_chars = min([nonlinear_chars, ceil(0.8*characteristics)])  # number of nonlinear characteristics
        m = np.zeros(nonlinear_chars)  # mean of taste parameter
        cov = np.eye(nonlinear_chars)  # covariance of taste parameter
        v = np.random.multivariate_normal(m, cov, size=population)  # random taste parameter

        # -- Product characteristics --#
        char_cor = np.random.rand(characteristics, characteristics)
        char_cor = np.dot(char_cor, char_cor.T)  # random correlation among product attributes
        chara = 2 + np.abs(np.dot(np.random.normal(size=(products, characteristics)),
                   np.linalg.cholesky(char_cor)))  # Product characteristics

        # -- Unbalanced panel --#
        obs = ceil(products * markets * (0.7 + 0.3 * np.random.rand()))
        X = np.empty((obs, characteristics))  # add two columns for price and market id
        M = np.zeros(obs, dtype=int)  # the market id vector
        counter = 0
        market_id = 0
        p = 1.0 / ceil(0.7 * products)
        min_prods = ceil(0.2 * products)
        while counter < obs:
            num_prods = max([min_prods, min(np.random.geometric(p), products)])
            num_prods = min([num_prods, obs - counter])
            selected = np.random.random_integers(0, products-1, num_prods)
            X[counter: (counter + num_prods), :] = chara[selected, :]
            M[counter: (counter + num_prods)] = market_id
            market_id += 1
            counter += num_prods

        # -- Product prices  -- #
        ml = np.zeros(obs)
        covl = np.eye(obs)
        P = 0.5 + np.random.exponential(4, size=obs) + \
            np.dot(X, np.abs(np.random.rand(characteristics)))
        Z = np.empty((obs, characteristics + 1))  # the instruments
        Z[:, 0] = 0.25 * P + 0.4 * np.random.rand(obs)
        Z[:, 1:] = X
        demand_shock = np.random.multivariate_normal(ml, covl) * 0.5
        P += demand_shock * 0.5

        # -- Market shares --#
        sigma = np.abs(np.random.rand(nonlinear_chars))
        v = sigma * v
        mu = np.dot(X[:, :nonlinear_chars], v.T)  # This creates a (obs x population) matrix of valuations
        alpha = - np.random.rand()
        beta = 3 * np.random.rand(characteristics - nonlinear_chars)
        common = alpha * P.reshape(-1, 1) + X[:, (nonlinear_chars):].dot(beta).reshape(-1, 1) + demand_shock.reshape(-1, 1)
        mu = np.exp(common + mu)

        S = np.zeros(obs)
        for j in range(population):
            sum_ex = np.bincount(M, weights=mu[:, j])
            S += mu[:, j] / (1 + sum_ex[M])
        S = S / population

        # -- Store -- #
        self.X1 = np.hstack((P.reshape(-1, 1), X[:, (nonlinear_chars):]))
        self.X2 = np.hstack((P.reshape(-1, 1), X[:, :nonlinear_chars]))
        self.Z = Z
        self.M = M
        self.S = S
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.demand_shock = demand_shock
        return


if __name__ == "__main__":
    market = Faker()
    market.genData(50, 15, 6, 500)
    blp = BLP(market.X1, market.X2, market.Z, market.M, market.S)
    blp.prepareSample()
    a = np.random.rand(market.X2.shape[1])
    ls = np.bincount(market.M, weights=market.S)
    s = ls[market.M]

    res = blp.solve(a, method="Nelder-Mead")

    print("The resulting message is : %s" % res.message)
    print("The flag is: %s" % res.success)
    print("Found theta1 is %s " % res.theta1)
    print("Real theta1 was %s %s" % (market.alpha, market.beta))
    print("Found theta2 is %s" % (res.theta2))
    print("Real theta2 was %s" % market.sigma)
    print("Starting theta2 was %s" % a)

    bounds = tuple([(-3, 3) for j in a])
    res = blp.solve(a, method="BFGS", bounds=bounds)

    print("The resulting message is : %s" % res.message)
    print("The flag is: %s" % res.success)
    print("Found theta1 is %s " % res.theta1)
    print("Real theta1 was %s %s" % (market.alpha, market.beta))
    print("Found theta2 is %s" % (res.theta2))
    print("Real theta2 was %s" % market.sigma)
    print("Starting theta2 was %s" % a)

    # [ 0.75817356  0.45750418  0.91629948  0.89341141  0.30887092]
