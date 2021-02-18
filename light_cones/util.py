import numpy as np
from numba import jit

# TODO:
# initialize kappa to what it would be if you never found anything in that pixel across all shells
# -1 force the fiducial to be 0.
# don't just add += 1 but add the weighted lenging kernel inv_rho_mean

@jit(nopython = True)
def histogram_hp(rho,heal):

    for i in range(len(heal)):
        ind = heal[i]
        rho[ind] += 1

    return rho

@jit(nopython = True)
def add_kappa_shell(kappa_i,rho_ij,rho_mean,r_s,rj,aj,drj):
    lensing_kernel = ((r_s-rj)*rj/(aj*r_s))*drj
    inv_rho_mean = 1./rho_mean
    for i in range(len(rho_ij)):
        kappa_i[i] += (rho_ij[i]*inv_rho_mean-1.)*lensing_kernel
    
    return kappa_i
