"""
Provides some standard sampling functions
"""

import numpy as np
import numpy.linalg as alg
import numpy.random as rnd

def uniform_sphere(N, d):
    """Samples uniformly from the sphere
        
        Args:
            N (int): number of samples
            d (int): dimension of space containing the sphere

        Returns:
            Z (float): array of shape (N,d) containing N independent samples 
                from the (d-1)-sphere
    """

    Z = rnd.normal(size=(N,d))
    return Z/alg.norm(Z, axis=1).reshape(-1,1)

def uniform_sphere_gen(N, d, gen):
    """Samples uniformly from the sphere using given RNG
        
        Args:
            N (int): number of samples
            d (int): dimension of space containing the sphere
            gen (rnd._generator.Generator): RNG to be used

        Returns:
            Z (float): array of shape (N,d) containing N independent samples 
                from the (d-1)-sphere generated with gen
    """

    Z = gen.normal(size=(N,d))
    return Z/alg.norm(Z, axis=1).reshape(-1,1)

def uniform_great_subsphere(p):
    """Samples uniformly from the great subsphere of the (d-1)-sphere w.r.t.
        the pole p
    
        Args:
            p (array): pole, should be 1d numpy array of norm 1

        Returns:
            y (array): 1d numpy array of same size as p, is a sample from 
                the great subsphere w.r.t. p
    """

    Z = rnd.normal(size=p.shape[0])
    Z = Z - np.inner(Z, p) * p
    return Z/alg.norm(Z)

def uniform_ball(N, d, b):
    """Samples uniformly from d-dimensional zero-centered Euclidean balls
    
        Args:
            N (int): number of samples
            d (int): dimension of space containing the ball
            b (float): radius of the ball
    """

    thetas = uniform_sphere(N, d)
    rs = b * rnd.uniform(size=(N,1))**(1/d)
    return rs * thetas

