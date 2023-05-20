"""
Provides some utility functions for the analysis of MCMC runs

Available Functions:
    remove_burn_in
    get_radii
    get_radii_list
    get_log_radii
    get_log_radii_list
    get_steps
    get_steps_list
    get_margs_list
    get_abs_margs_list
    log_den_vals
    mss
    mss_list
    acf
    iat
    iat_list
    n_eff
"""

import numpy as np
import numpy.linalg as alg

def remove_burn_in(sam_list, burn_in):
    """Removes a given burn-in from a list of sets of samples

        Args:
            sam_list: list of 2d np arrays containing samples
            burn_in: a non-negative integer giving the number of iterations used
                for the burn-in phase

        Returns:
            burned_list: list containing the samples from sam_list with the 
                burn-in iterates removed
    """
    return [samples[burn_in:] for samples in sam_list]

def get_radii(samples):
    """Computes the radii (Euclidean norms) of given samples

        Args:
            samples: 2d np array of shape (nsamples, d) where samples[i] is the
                i-th sample

        Returns:
            radii: np array of size nsamples where radii[i] is the radius of
                samples[i]
    """
    return alg.norm(samples, axis=1)

def get_radii_list(sam_list):
    """Computes the radii (Euclidean norms) of given samples

        Args:
            sam_list: list of 2d np arrays containing samples

        Returns:
            radii_list: list of 1d np arrays containing the radii of the given
                samples
    """
    return [get_radii(samples) for samples in sam_list]

def get_log_radii(samples):
    """Computes the logarithms of radii (Euclidean norms) of given samples

        Args:
            samples: 2d np array of shape (nsamples, d) where samples[i] is the
                i-th sample

        Returns:
            log_radii: np array of size nsamples where log_radii[i] is the 
                natural logarithm of the radius of samples[i]
    """
    return np.log(get_radii(samples))

def get_log_radii_list(sam_list):
    """Computes the logarithms of radii (Euclidean norms) of given samples

        Args:
            sam_list: list of 2d np arrays containing samples

        Returns:
            log_radii_list: list of 1d np arrays containing the log radii of the
                given samples
    """
    return [get_log_radii(samples) for samples in sam_list]

def get_steps(samples):
    """Computes the empirical step sizes taken by a sampler, i.e. the Euclidean 
        distances between each pair of consecutive samples

        Args:
            samples: 2d np array of shape (nsamples, d) where samples[i] is the
                i-th sample

        Returns:
            steps: np array of size nsamples-1 where steps[i] is the Euclidean
                distance between samples[i] and samples[i+1]
    """
    return alg.norm(samples[1:] - samples[:-1], axis=1)

def get_steps_list(sam_list):
    """Computes the empirical step sizes for given samples

        Args:
            sam_list: list of 2d np arrays containing samples

        Returns:
            steps_list: list of 1d np arrays containing the step sizes for the
                given samples
    """
    return [get_steps(samples) for samples in sam_list]

def get_margs_list(sam_list, i_mar):
    """Extracts univariate marginal samples from a list of sets of samples

        Args:
            sam_list: list of 2d np arrays containing samples
            i_mar: index of the marginal that is to be extracted

        Returns:
            marg_sam_list: list of 1d np arrays, where marg_sam_list[i_sam] is
                precisely sam_list[i_sam][:,i_mar]
    """
    return [samples[:,i_mar] for samples in sam_list]

def get_abs_margs_list(sam_list, i_mar):
    """Extracts absolute values of univariate marginal samples from a list of 
        sets of samples

        Args:
            sam_list: list of 2d np arrays containing samples
            i_mar: index of the marginal whose absolute values are to be extracted

        Returns:
            abs_marg_sam_list: list of 1d np arrays, where marg_sam_list[i_sam] is
                precisely np.abs(sam_list[i_sam][:,i_mar])
    """
    return [np.abs(samples[:,i_mar]) for samples in sam_list]

def log_den_vals(samples, log_density):
    """Computes log density values of given samples

        Args:
            samples: 2d np array of shape (nsamples, d) where samples[i] is the
                i-th sample
            log_density: logarithm of a potentially unnormalized probability 
                density such that samples[i] is a valid argument for log_density
                for all i

        Returns:
            log_den_vals: 1d np array containing log density values of the given
                samples, i.e. log_den_vals[i] = log_density(samples[i])
    """
    return np.array(list(map(log_density, samples)))

def mss(samples):
    """Computes the mean step size of given samples

        Args:
            samples: 2d np array of shape (nsamples, d) where samples[i] is the
                i-th sample

        Returns:
            mss: a single float giving the mean step size of the samples
    """
    return np.mean(get_steps(samples))

def mss_list(sam_list):
    """Computes the mean step size of given samples for several instances given
        as a list

        Args:
            sam_list: list of 2d np arrays containing samples

        Returns:
            mss_list: 1d np array, where mss_list[i] is the MSS of the samples
                in sam_list[i] 
    """
    return np.array([mss(samples) for samples in sam_list])

def acf(vals, maxl=None):
    """Computes autocorrelation function
        
        Args:
            vals: values to compute acf of, should be 1d np array
            maxl: maximum lag for which acf is to be computed, the default is
                vals.shape[0]//2

        Returns:
            autocors: vector of size maxl+1 containing the autocorrelations for
                all lags from 0 to maxl
    """
    nvals = vals.shape[0]
    if maxl == None:
        maxl = nvals//2
    cvals = vals - np.mean(vals)
    autocov = lambda l: np.inner(cvals[l:], cvals[:nvals-l]) / nvals
    autocovs = np.array(list(map(autocov, range(maxl+1))))
    return autocovs/autocovs[0]

def iat(vals, maxl=None):
    """Computes the integrated autocorrelation time for given values by the
        heuristics described in Gelman et al "Bayesian Data Analysis", 
        Chapter 11.5

        Args:
            vals: values to compute IAT for, should be 1d np array
            maxl: largest lag between iterations to consider for the result,
                the default is vals.shape[0]//2

        Returns:
            iat: a positive float giving the IAT for vals 
    """
    n = vals.shape[0]
    if maxl == None:
        maxl = n//2
    ac = acf(vals, maxl)
    sums = ac[2:maxl-(maxl % 2):2] + ac[3::2]
    inds = np.arange(sums.shape[0])[sums < 0]
    L = 1 + 2 * inds[0] if inds.shape[0] > 0 else n-1
    return 1.0 + np.max([2 * np.sum(ac[1:L+1]), 0.0])

def iat_list(val_list, maxl=None):
    """Computes the IAT of given values for several instances given as a list

        Args:
            val_list: list of 1d np arrays containing values
            maxl: largest lag between iterations to consider for the result,
                the default is half the length of the value array considered

        Returns:
            iat_list: 1d np array, where iat_list[i] is the IAT of the values in
                val_list[i] 
    """
    return np.array([iat(vals, maxl) for vals in val_list])

def n_eff(vals, maxl=None):
    """Computes effective sample size n_eff for given values

        Args:
            vals: values to compute n_eff for, should be 1d np array
            maxl: largest lag between iterations to consider for the result,
                the default is vals.shape[0]//2

        Returns:
            n_eff: a single float giving the effective sample size for vals
    """
    return vals.shape[0] / iat(vals, maxl)

