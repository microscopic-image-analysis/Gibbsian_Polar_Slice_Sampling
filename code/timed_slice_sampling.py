"""
Provides implementations of several efficient slice sampling algorithms where
the cumulative runtime is stored after each iteration and these times are
returned together with the samples in the end

Functions implementing the samplers:
    gibbsian_polar_ss
    hit_and_run_uniform_ss
    elliptical_ss

Convenience functions for easier use of elliptical_ss:
    log_prior
    log_prior_indep
    log_prior_identity
"""

import numpy as np
import numpy.linalg as alg
import numpy.random as rnd
import scipy.stats as sts
import standard_sampling_functions as ssf
import time as ti
from fastprogress import progress_bar

###################### Miscellaneous Auxiliary Functions #######################

def nrange(itnum, bar):
    """Auxiliary function, not to be called by the user"""

    if bar:
        return progress_bar(range(1, itnum+1))
    return range(1, itnum+1)

def standard_basis(d, i):
    """Auxiliary function, returns i-th element (w.r.t. zero-based indexation 
        of standard basis of \R^d
        
        Note: We don't use np.eye here because it appears to be unreasonably
        much slower than what we do instead.
    """

    v = np.zeros(d)
    v[i] = 1.0
    return v

def log_prior(cov, x):
    """Convenience function to compute values of the log prior density for
        the ESS

        Args:
            cov: covariance matrix for the mean zero Gaussian prior, should 
                be two-dimensional np array of shape (d,d) representing a 
                positive definite matrix
            x: location at which the log prior density is to be evaluated, 
                should be one-dimensional np array of size d

        Returns:
            p(x): the value at x of the log density of a Gaussian with mean 
            zero and covariance matrix cov
    """

    return sts.multivariate_normal(np.zeros(x.shape[0]), cov).logpdf(x)

def log_prior_indep(var, x):
    """Convenience function to compute values of the log prior density for 
        the ESS when the prior's covariance matrix is diagonal

        Args:
            var: diagonal entries for the mean zero Gaussian prior, should be
                one-dimensional np array of size d containing positive floats
            x: location at which the log prior density is to be evaluated, 
                should be one-dimensional np array of size d

        Returns:
            p(x): the value at x of the log density of a Gaussian with mean 
            zero and covariance matrix diag(var)
        
        Runtime: O(d)
    """

    d = x.shape[0]
    log_constant = -d/2 * np.log(2*np.pi) -1/2 * np.sum(np.log(var))
    return log_constant - 1/2 * np.inner(x, var**(-1) * x)

def log_prior_identity(var, x):
    """Convenience function to compute values of the log prior density for 
        the ESS when the prior's covariance matrix is a rescaling of the 
        identity matrix

        Args:
            var: positive float such that var*np.identity(d) is the covariance
                matrix of the prior
            x: location at which the log prior density is to be evaluated, 
                should be one-dimensional np array of size d

        Returns:
            p(x): the value at x of the log density of a Gaussian with mean 
            zero and covariance matrix var*np.identity(d)
        
        Runtime: O(d)
    """

    return log_prior_indep(var*np.ones(x.shape[0]), x)

############################## Sampling Functions ##############################

def orbit_shrinkage(log_trafo, r_old, p, log_t, y):
    """Auxiliary function, not to be called by the user"""

    omega = rnd.uniform(0,2*np.pi)
    omega_min = omega - 2*np.pi
    omega_max = omega
    theta = p * np.cos(omega) + y * np.sin(omega)
    while log_trafo(r_old, theta) <= log_t:
        omega = rnd.uniform(omega_min, omega_max)
        theta = p * np.cos(omega) + y * np.sin(omega)
        theta = theta / alg.norm(theta) # prevents accumulating errors
        if omega < 0:
            omega_min = omega
        else:
            omega_max = omega
    return theta

def radius_shrinkage(log_trafo, w, r_old, log_t, theta):
    """Auxiliary function, not to be called by the user"""

    # determine initial interval via stepping out from r_old
    ali = rnd.uniform()
    r_min, r_max = ( r_old - ali * w, r_old + (1-ali) * w )
    r_min = np.max([r_min, 0])
    while r_min > 0 and log_trafo(r_min, theta) > log_t:
        r_min = np.max([r_min - w, 0])
    while log_trafo(r_max, theta) > log_t:
        r_max += w
    # sample new radius via shrinkage around r_old
    r = rnd.uniform(r_min, r_max)
    while log_trafo(r, theta) <= log_t:
        if r < r_old:
            r_min = r
        else:
            r_max = r
        r = rnd.uniform(r_min, r_max)    
    return r

def gibbsian_polar_ss(log_density, x_0, w, itnum, bar=True):
    """Runs the Gibbsian Polar Slice Sampler, using shrinkage geodesic slice
        sampling for the direction update and stepping-out and shrinkage
        for the radius update

        Args:
            log_density: log of the target density
            x_0: initial value, should be size d np array describing a point 
                from the support of the target density
            w: initial width of the radius search area, must be positive and
                choosing it overly small can impair the sampler's efficiency
            itnum: number of iterations to run the algorithm for, should be non-
                negative integer
            bar: bool denoting whether or not a progress bar should be used

        Returns:
            samples: np array of shape (itnum+1, d), where samples[i] is the 
                i-th sample generated by the GPSS
    """

    # auxiliary defs, see footnote (1)
    d = x_0.shape[0]
    log_trafo = lambda r, theta: (d-1)*np.log(r) + log_density(r*theta)
    # generate chain
    time = np.zeros(itnum+1)
    t0 = ti.time()
    R = np.zeros(itnum+1)
    Theta = np.zeros((itnum+1,d))
    R[0] = alg.norm(x_0)
    Theta[0] = x_0 / R[0]
    for n in nrange(itnum, bar):
        log_t = log_trafo(R[n-1], Theta[n-1]) + np.log(rnd.uniform())
        y = ssf.uniform_great_subsphere(Theta[n-1])
        Theta[n] = orbit_shrinkage(log_trafo, R[n-1], Theta[n-1], log_t, y)
        R[n] = radius_shrinkage(log_trafo, w, R[n-1], log_t, Theta[n])
        time[n] = ti.time() - t0
    X = R.reshape(-1,1) * Theta
    return X, time

def hit_and_run_uniform_ss(log_density, x_0, w, itnum, bar=True):
    """Runs the Hit-and-run Uniform Slice Sampler, which in every iteration
        chooses a direction uniformly at random and performs a univariate slice
        sampling step with stepping out and shrinkage on the line through the 
        previous sample in this direction

        Args:
            log_density: log of the target density, should take one-dimensional 
                np arrays of size d as input and output floats
            x_0: initial value, should be size d np array describing a point 
                from the support of the target density
            w: initial width of the update search area, must be positive and
                choosing it overly small can impair the sampler's efficiency
            itnum: number of iterations to run the algorithm for, should be non-
                negative integer
            bar: bool denoting whether or not a progress bar should be used

        Returns:
            samples: np array of shape (itnum+1, d), where samples[i] is the 
                i-th sample generated by the HRUSS
    """

    d = x_0.shape[0]
    time = np.zeros(itnum+1)
    t0 = ti.time()
    X = np.zeros((itnum+1, d))
    X[0] = x_0
    for n in nrange(itnum, bar):
        log_t = log_density(X[n-1]) + np.log(rnd.uniform())
        ali = rnd.uniform()
        L, R = ( -ali * w, (1-ali) * w )
        v = ssf.uniform_sphere(1,d).reshape(d)
        # stepping out
        while log_density(X[n-1] + L*v) > log_t:
            L -= w
        while log_density(X[n-1] + R*v) > log_t:
            R += w
        # sampling and shrinking
        alpha = rnd.uniform(L, R)
        x_prop = X[n-1] + alpha * v
        while log_density(x_prop) <= log_t:
            if alpha < 0:
                L = alpha
            else:
                R = alpha
            alpha = rnd.uniform(L, R)
            x_prop = X[n-1] + alpha * v
        X[n] = x_prop
        time[n] = ti.time() - t0
    return X, time

def elliptical_ss(cov, log_likelihood, x_0, itnum, bar=True):
    """Runs the Elliptical Slice Sampler

        Args:
            cov: specifies covariance matrix for the mean zero Gaussian prior,
                there are three ways to use this argument: 1. set it to a
                two-dimensional np array of shape(d,d) that represents a 
                positive definite matrix, which is then used as the covariance
                matrix, 2. set it to a one-dimensional np array of size d, the
                covariance matrix used is then diag(cov), 3. set it to a float, 
                the covariance matrix used is then cov * np.identity(d)
            log_likelihood: log of the likelihood-part of the target posterior,
                should take one-dimensional np arrays of size d as input and
                output floats
            x_0: initial value, should be size d np array describing a point 
                from the support of the target density
            itnum: number of iterations to run the algorithm for, should be non-
                negative integer
            bar: bool denoting whether or not a progress bar should be used

        Returns:
            samples: np array of shape (itnum+1, d), where samples[i] is the 
                i-th sample generated by the ESS
    """

    d = x_0.shape[0]
    sample_y = None
    if type(cov) == float:
        sample_y = lambda: rnd.normal(0.0, np.sqrt(cov), d)
    elif len(cov.shape) == 1:
        sample_y = lambda: np.sqrt(cov) * rnd.normal(0.0, 1.0, d)
    else:
        sample_y = lambda: rnd.multivariate_normal(np.zeros(d), cov)
    time = np.zeros(itnum+1)
    t0 = ti.time()
    X = np.zeros((itnum+1, d))
    X[0] = x_0
    for n in nrange(itnum, bar):
        log_t = log_likelihood(X[n-1]) + np.log(rnd.uniform())
        y = sample_y()
        omega = rnd.uniform(0, 2*np.pi)
        omega_min, omega_max = ( omega - 2*np.pi, omega )
        x_prop = X[n-1] * np.cos(omega) + y * np.sin(omega)
        while log_likelihood(x_prop) <= log_t:
            if omega < 0:
                omega_min = omega
            else:
                omega_max = omega
            omega = rnd.uniform(omega_min, omega_max)
            x_prop = X[n-1] * np.cos(omega) + y * np.sin(omega)
        X[n] = x_prop
        time[n] = ti.time() - t0
    return X, time

################################## Footnotes ###################################
#                                                                              #
# (1):  The function log_trafo takes radius and direction as separate args     #
#       because the argument to be used for it is usually already available    #
#       in this decomposition and using a compound argument would necessitate  #
#       computing its norm to compute the function, which is unnecessarily     #
#       slow (in reasonably high dimensions).                                  #
#                                                                              #
################################################################################

