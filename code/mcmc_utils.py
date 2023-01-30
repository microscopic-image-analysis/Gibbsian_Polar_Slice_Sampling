"""
Provides some utility functions for the analysis of MCMC runs
"""

import numpy as np

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

def IAT(vals, maxl=None):
    """Computes the integrated autocorrelation time for given values by the
        heuristics described in Gelman et al "Bayesian Data Analysis", 
        Chapter 11.5

        Args:
            vals: values to compute IAT for, should be 1d np array
            maxl: largest lag between iterations to consider for the result,
                the default is vals.shape[0]//2

        Returns:
            IAT: a positive float giving the IAT for vals 
    """

    n = vals.shape[0]
    if maxl == None:
        maxl = n//2
    ac = acf(vals, maxl)
    sums = ac[2:maxl-(maxl % 2):2] + ac[3::2]
    inds = np.arange(sums.shape[0])[sums < 0]
    L = 1 + 2 * inds[0] if inds.shape[0] > 0 else n-1
    return 1.0 + np.max([2 * np.sum(ac[1:L+1]), 0.0])

def n_eff(vals, maxl=None):
    """Computes effective sample size n_eff for given values

        Args:
            vals: values to compute n_eff for, should be 1d np array
            maxl: largest lag between iterations to consider for the result,
                the default is vals.shape[0]//2

        Returns:
            n_eff: a single float giving the effective sample size for vals
    """

    return vals.shape[0] / IAT(vals, maxl)

