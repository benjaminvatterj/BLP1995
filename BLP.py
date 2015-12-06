import numpy as np
from scipy.optimize import minimize
from sys import stdout
from time import time
import multiprocessing
from itertools import repeat


class BLPException(Exception):

    """ A basic exception class """
    pass


class BLPResult:

    """ A class to hold the results

    Attributes:
    ----------
    res : OptimizeResult
        the original Scipy optimizer results
    theta1 : ndarray
        the linear coefficients of demand
    theta2 : ndarray
        the non-linear coefficients of demand
    success : Boolean
        the success status of the optimization
    message : string
        the solver's exit message
    """

    def __init__(self, res, theta1, theta2):
        self.res = res
        self.theta1 = theta1
        self.theta2 = theta2
        self.success = res.success
        self.message = res.message


class BLP:

    """
    A BLP(1995) implementation
    based upon Aviv Nevo's Practitioner's
    Guide. It includes a derivative-free method,
    Nelder-Mead, and a bounded gradient method,
    L-BFGS-B. The choice of the bounded version
    of the BFGS method was made as to avoid
    fine-tuning the step size, that might lead
    to undefined region and problems in convergence.
    The recommended procedure is to first run a
    non-derivative method with medium precision, and
    later execute a bounded derivative with high precision
    on a bounded region.

    Arguments:
    ---------
    X1 : ndarray
         variables that enter the problem linearly,
         such as price and brand dummy variables
    X2 : ndarray
         variables that enter the problem nonlinearly,
         such as price and product characteristics
    Z  : ndarray
         Instrument matrix for the X1 matrix.
    M  : ndarray
         A vector identifying the market of each
         observation in the X1, X2 and Z data.
    S  : ndarray
         The vector of market shares
    weight : ndarray, optional
         the weight matrix (Phi in Nevo's guide)
         to use for the objective function. Defaults
         to Z'Z.
    parallel : boolean, optional
         Whether to parallelize the delta fixed point
         by a set of markets. Defaults to False.
    threads : int, optional
         Number of cores to use in parallelization.
         If parallel is False it doesn't matter, if
         set to None and parallel is True it defaults
         to the max available cpus.
    par_cut : int/string, optional
        The number of cuts to perform on the markets
        for the delta fixed point. If a number n is provided
        there will be n fixed point iterations per optimizer step.
        If the string "clean" is submitted then there will be as
        many cuts as cores used.
    """

    status_string = {
        0: "Initiated, population not sampled.",
        1: "Initiated, population sampled, results not done.",
        2: "Initiated, populations sampled, results available."
    }

    def __init__(self, X1, X2, Z, M, S, weight=None, parallel=False,
                 threads=None, par_cut="clean"):
        self.status = 0  # initial status for the class
        self.X1 = X1
        self.X2 = X2
        self.Z = Z
        self.M = M
        if not np.all(np.sort(M) == M):
            raise BLPException("Data must be ordered by market id previously")
        self.S = np.log(S)  # store the log as use this more
        self.population_size = 0
        self.delta_tol = 1e-8
        self.max_delta_loop = 1e6
        self.delta_method = 0  # default to Picard iterations
        self.iterations = 0
        self.adaptive_delta_tol = False
        self.verbose = True
        self.v = 0  # store the value function
        self.prev_v = 0  # the previous value function
        self.prev_x = []  # the previous theta2 vector
        self.jac_required = False
        # v: taste parameters, D: demographics
        self.population = {"v": [], "D": []}

        # Sum the observed market shares by market
        sum_shares = np.bincount(M, weights=S)
        # Calculate the outside good share
        outside_share = 1 - sum_shares[M]
        # Use this to set the initial delta
        self.initial_delta = np.log(S) - np.log(outside_share)

        # parallelizing setting
        self.parallel = parallel

        uq = np.unique(self.M)
        if parallel:
            if threads is None:
                threads = multiprocessing.cpu_count()
            elif threads > multiprocessing.cpu_count():
                raise BLPException("Can not use more cpus than available (%d found, %d requested)" % (multiprocessing.cpu_count(), threads))

            self.threads = threads  # set threads
            if par_cut == "clean":
                par_cut = threads

            self.chunksize = par_cut // threads
        else:
            if par_cut == "clean":
                par_cut = 1

        if isinstance(par_cut, int):
            j = 0
            step = len(uq) // par_cut
            self.par_cuts = []
            while j < len(uq):
                start = np.nonzero(self.M == uq[j])[0][0]
                if len(uq) - j - step < step:
                    step = len(uq) - j
                end = np.nonzero(self.M == uq[j + step - 1])[0][-1]
                self.par_cuts.append([start, end + 1])
                j += step
        else:
            raise BLPException("par_cut is neither <<clean>> or an integer.")

        # Prepare the projection from theta2 to theta1
        self.prepareProjection(weight)

        # Prepare the strainer for the jacobian
        self.strainer = np.zeros((X1.shape[0], X1.shape[0]), dtype=bool)
        u, q = np.unique(self.M, return_counts=True)
        pos = 0
        for i in q:
            self.strainer[pos:pos+i, pos:pos+i] = True
            pos += i

        return

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def prepareProjection(self, weight=None):
        """
        Prepares the projection from theta2
        to theta1:
        (X1'Z Phi^{-1} Z' X1)^{-1}X1'Z Phi^{-1}
        And also stores
        Phi^{-1}
        for the objective function

        Argument:
        --------
        weight : ndarray, optional
                 the weight matrix (Phi in Nevo's guide)
                 to use for the objective function. Defaults
                 to Z'Z.

        Notes:
        ------
        This is left "public" to allow for a weight matrix
        update if desired. Also the structure of the projection
        is chosen to reduce the amount of memory required, as
        adding the Z matrix that follow will enlarge the number
        of column to the sample size, instead of being the
        number of instruments.
        """
        if weight is None:
            weight = self.Z.T.dot(self.Z)
        # f <- Phi^{-1} Z'X1
        f = np.linalg.solve(weight, self.Z.T.dot(self.X1))
        # (X1'Xf)^{-1}f'
        self.projection = np.linalg.solve(self.X1.T.dot(self.Z).dot(f), f.T)
        self.phi_inverse = np.linalg.inv(weight)
        return

    def prepareSample(self, PV=None, PD=None, population=500):
        """
        Samples the population for the problem

        Arguments:
        ----------
        PV : callable, optional
             The distribution of the taste parameters.
             Must accept a size keyword argument. Results of this
             function must follow the same of shape as the multivariate
             distributions available in Numpy. Defaults to standard
             multivariate normal.

        PD : callable, optional
             The distribution of demographic. Must accept
             a size keyword argument. Results of this
             function must follow the same of shape as the multivariate
             distributions available in Numpy.
             Defaults to no demographics.

        population : int, optional
             Size of the population to sample.
        """
        if PV is None:
            m = np.zeros(self.X2.shape[1])
            c = np.eye(self.X2.shape[1])
            self.population["v"] = np.random.multivariate_normal(m, c, population)
        else:
            self.population["v"] = PV(size=population)

        if PD is not None:
            self.population["D"] = PD(size=population)

        self.population_size = population
        self.status = 1
        return

    def _deltas(self, theta2, bounds=[0, None]):
        """
        Solves the delta fixed point
        using theta1 and theta2.

        Argument:
        --------
        theta2 : ndarray
                 nonlinear demand coefficients.
                 Note: this vector first contains
                 the K taste coefficients later the
                 KxD demographic-characteristics
                 coefficients where these are ordered
                 first by characteristic, i.e:
                 [d1k1 d1k2 d1k3 d2k1 d2k2 d2k3]

        bounds : list, optional
                The starting and ending point in the data to consider
                (used for parallelization).

        Returns:
        -------
            out : ndarray
                 vector of mean valuations
        """
        start = bounds[0]
        end = bounds[1]

        if end is None:
            end = self.X2.shape[0]

        # 1) calculate the populations individual preferences
        v = self.population["v"] * theta2[:self.X2.shape[1]]
        if self.population["D"] is not []:
            D = np.dot(self.population["D"], theta2[self.X2.shape[1]:].reshape(-1, self.X2.shape[1]))
        else:
            D = 0
        # This is now a matrix of dimensions N x population
        x = self.X2[start:end, :].dot((v + D).T)

        # Preallocate st
        st = np.zeros(end - start)

        # 2) Loop on delta until convergence
        distance = 1
        delta = self.initial_delta[start:end]
        it = 0  # never loop with no breaks

        if self.delta_method == 1:
            anderson = np.empty((6, end - start))  # Storage for Anderson (m=3)
            prev_delta = delta.copy()
            prev_diff = delta.copy()

        while (distance > self.delta_tol) and (it < self.max_delta_loop):

            # 2.1) Estimate market shares
            ex = np.exp(delta.reshape(-1, 1) + x)
            st.fill(0)  # clean

            for j in range(self.population_size):
                sum_ex = np.bincount(self.M[start:end], weights=ex[:, j])
                st += ex[:, j] / (1 + sum_ex[self.M[start:end]])

            st = st / self.population_size

            # 2.2) Get the next delta
            delta_t = delta + self.S[start:end] - np.log(st)
            diff = delta_t - delta

            # Anderson Acceleration
            if self.delta_method == 1:

                if it > 0:

                    anderson[it % 3, :] = diff - prev_diff
                    anderson[3 + (it % 3), :] = delta_t - prev_delta
                    np.copyto(prev_delta, delta_t)
                    np.copyto(prev_diff, diff)

                    if it > 2:
                        Fk = anderson[:3, :]
                        Xk = anderson[3:, :]
                        alphas = np.linalg.lstsq(Fk.T, diff)[0]
                        delta_t = delta_t - Xk.T.dot(alphas)
                        diff = delta_t - delta
                else:
                    np.copyto(prev_delta, delta_t)
                    np.copyto(prev_diff, diff)

            if not np.all(~np.isnan(delta_t)):
                print(delta_t)
                print(self.initial_delta)
                raise BLPException("undefined delta on iteration %d" % it)

            # 2.3) Calculate distance (infinite norm)
            distance = np.max(np.abs(diff))

            # 2.4) Update delta
            np.copyto(delta, delta_t)

            it += 1
            if it >= self.max_delta_loop:
                raise BLPException("Delta fixed point failed to converge.")

        # print("Delta convergence took %d iterations" % it)

        if not self.jac_required:
            return delta_t
        else:
            dSdDelta = np.zeros((end-start, end-start))
            dSdTheta = np.zeros((end-start, theta2.shape[0]))

            for j in range(self.population_size):
                sum_ex = np.bincount(self.M[start:end], weights=ex[:, j])
                s = ex[:, j] / (1 + sum_ex[self.M[start:end]])
                dSdDelta += self.strainer[start:end, start:end] * (np.diag(s) - np.dot(s, s.T))

                L = s.reshape(-1, 1) * self.X2[start:end, :]
                for k in range(self.X2.shape[1]):
                    sum_sx = np.bincount(self.M[start:end], weights=L[:, k])
                    L[:, k] = L[:, k] - s * sum_sx[self.M[start:end]]

                dSdTheta[:, :self.X2.shape[1]] += L * self.population["v"][j, :]
                if theta2.shape[0] > self.X2.shape[1]:
                    pos = self.X2.shape[1]
                    dem_size = self.population["D"].shape[1]
                    for l in range(dem_size):
                        dSdTheta[:, pos:(pos + dem_size)] += L * self.population["D"][j, l]

            ddelta = -1.0 * np.linalg.solve(dSdDelta, dSdTheta)
            return (delta_t, ddelta)

    def _adapt_delta_tol(self):
        """
        Adapts the delta fixed point
        tolerance if configured to do so.
        It will reduce the tolerance every
        100 iterations up to the minimum
        between 10^{-3} times the objective
        function tolerance and 10^{-14}.
        """
        if ((self.adaptive_delta_tol and
             (self.iterations > 0) and
             (self.iterations % 100 == 0) and
             self.delta_tol > np.min(1e-14, self.ftol * 1e-3))):
            self.delta_tol *= 1e-1
        return

    def _objectiveFunction(self, theta2):
        """
        This is the objective function for
        the minimizer.

        Arguments:
        ----------
        theta2 : ndarray
                 the nonlinear cost parameters

        Returns:
        --------
        out : float
              objective function value.
        """
        if self.parallel:
            res = self.pool.starmap(self._deltas, zip(repeat(theta2), self.par_cuts), self.chunksize)
            if self.jac_required:
                delta = np.hstack([x[0] for x in res])
                ddelta = np.vstack([x[1] for x in res])
            else:
                delta = np.hstack(res)
        else:
            if len(self.par_cuts) == 1:
                if not self.jac_required:
                    delta = self._deltas(theta2)
                else:
                    delta, ddelta = self._deltas(theta2)
            else:
                delta = np.zeros(self.X1.shape[0])
                ddelta = np.zeros((self.X1.shape[0], self.X2.shape[1]))

                for j in self.par_cuts:
                    if not self.jac_required:
                        delta[j[0]:j[1]] = self._deltas(theta2, j)
                    else:
                        delta[j[0]:j[1]], ddelta[j[0]:j[1], :] = self._deltas(theta2, j)

        # Replace initial delta
        np.copyto(self.initial_delta, delta)

        theta1 = self.projection.dot(self.Z.T.dot(delta))
        Zomega = self.Z.T.dot(delta - self.X1.dot(theta1))
        self.iterations += 1
        self._adapt_delta_tol()
        pz = self.phi_inverse.dot(Zomega)
        self.v = Zomega.T.dot(pz)

        if not self.jac_required:
            # print("%s => %s" % (theta2, self.v))
            if np.isnan(self.v):
                self.v = np.inf
            return self.v
        else:
            dv = 2.0 * ddelta.T.dot(self.Z).dot(pz)
            # print("%s => %s %s" % (theta2, self.v, dv))
            stdout.flush()
            return (self.v, dv)

        return

    def _iter_print(self, xk):
        if self.verbose:
            if self.iterations <= 1:
                print("iteration      x-step         f-step      objective      time     ")
            else:
                print("%8d    %5e    %5e   %5e    %5e" % (self.iterations,
                      np.max(np.abs(xk - self.prev_x)),
                      self.v - self.prev_v,
                      self.v,
                      time() - self.iter_timer))
            self.prev_v = self.v
            self.prev_x = xk
            self.iter_timer = time()
            stdout.flush()
        return

    def solve(self, initial_theta2, method="Nelder-Mead",
              ftol=1e-8, xtol=1e-8, maxiter=1e5, maxfev=1e5,
              bounds=None, delta_tol=1e-7, delta_max_iter=1e4,
              adaptive_delta_tol=True, delta_method="picard",
              verbose=True):
        """
        Runs the BLP estimation procedure.

        Arguments:
        ----------
        intial_theta2 : ndarray
                starting values for the nonlinear
                demand parameters.
        method : string, optional
                The minimizer to use. Should be
                either "Nelder-Mead" or "BFGS".
                Defaults to Nelder-Mead non-derivative
                simplex method.
        ftol : float, optional
                tolerance of the objective function
        xtol : float, optional
                Relative error in solution acceptable for convergence
        maxiter : int, optional
                maximum number of iteration of the
                objective function
        maxfev : int, optional
                maximum number of function evaluations
                to make.
        bounds : tuple, optional
                The bounds to use in the L-BFGS-B
                algorithm.
        delta_tol : float, optional
                tolerance of delta fixed point
        delta_max_iter : int, optional
                maximum number of iterations of the
                delta fixed point
        adaptive_delta_tol : boolean, optional
                whether to gradually reduce the
                tolerance of delta as the iterations
                increase.
        method : string, optional
                Convergence method. Default is picard,
                which is the standard BLP method.
                Other option is "anderson" which uses
                Anderson acceleration.
        verbose : boolean, optional
                Whether to print the iteration
                steps while solving.

        Returns:
        --------
        res : BLPResult
            The optimization results.
        """
        if self.status == 0:
            raise BLPException(self.status_string[self.status])

        if delta_method not in ["picard", "anderson"]:
            raise BLPException("delta method %s unknown" % delta_method)
        else:
            if delta_method == "picard":
                self.delta_method = 0
            else:
                self.delta_method = 1

        self.iterations = 0
        self.delta_tol = delta_tol
        self.adaptive_delta_tol = adaptive_delta_tol
        self.ftol = ftol
        self.prev_x = initial_theta2
        self.max_delta_loop = delta_max_iter
        self.iter_timer = time()
        self.verbose = verbose
        self._iter_print(self.prev_x)

        if self.parallel:
            self.pool = multiprocessing.Pool(self.threads)

        if method == "Nelder-Mead":
            res = minimize(self._objectiveFunction, initial_theta2,
                           method="Nelder-Mead", tol=ftol,
                           options={
                            'maxiter': maxiter,
                            'xtol': xtol,
                            'ftol': ftol,
                            'maxfev': maxfev
                            },
                           callback=self._iter_print)
            delta = self._deltas(res.x)
        elif method == "BFGS":
            if bounds is None:
                raise BLPException("Bounds must be set for the BFGS algorithm.")
            self.jac_required = True
            res = minimize(self._objectiveFunction, initial_theta2,
                           method="L-BFGS-B", tol=ftol, jac=True,
                           options={
                            'maxiter': maxiter,
                            'gtol': ftol,
                           },
                           bounds=bounds,
                           callback=self._iter_print)
            delta, ddelta = self._deltas(res.x)
        else:
            raise BLPException("Unknown solver choice.")

        if self.parallel:
            self.pool.close()

        theta1 = self.projection.dot(self.Z.T.dot(delta))
        results = BLPResult(res, theta1, res.x)
        return results
