import os
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import asdf
import glob
import gc
import time
from scipy.interpolate import interp1d
from classy import Class

# cosmological parameters
h = 0.6736
H0 = h*100.# km/s/Mpc
Om_m = 0.315192
c = 299792.458# km/s

# healpix parameters
nside = 16384
delta_omega = hp.nside2pixarea(nside)
npix = (hp.nside2npix(nside))

# if downsampling
nside_out = 4096
npix_out = (hp.nside2npix(nside_out))
delta_omega_out = hp.nside2pixarea(nside_out)

# simulation name
simname = "/mnt/store/lgarrison/AbacusSummit_base_c000_ph006/lightcones/heal/"
Lbox = 2000. # Mpc/h
PPD = 6912
NP = PPD**3

# particle density in 1/Mpc^3
n_part = NP/(Lbox/h)**3

# all snapshots and redshifts that have light cones; early to recent redshifts
zs_all = np.load("data_headers/redshifts.npy")

# ordered from small to large; small step number to large step number
steps_all = np.load("data_headers/steps.npy")

# comoving distances in Mpc/h; far shells to close shells
chis_all = np.load("data_headers/coord_dist.npy")
chis_all /= h # Mpc

# furthest and closest shells are at what time step
step_min = np.min(steps_all)
step_max = np.max(steps_all)

# location of the observer
origin = np.array([-990.,-990.,-990.])

# distance from furthest point to observer in Mpc/h
chi_max = 1.5*Lbox-origin[0]
chi_max /= h # Mpc

# select the final and initial step for computing the convergence map
step_start = steps_all[np.argmax((chis_all-chi_max) < 0)]# corresponds to 4000-origin
print("starting step = ",step_start)
print("furthest redshift = ",zs_all[np.argmax((chis_all-chi_max) < 0)])
step_stop = step_max

# CMB information, computed using CLASS
z_cmb = 1089.276682
chi_cmb = 13872.661199427605 # Mpc 
print("distance to CMB = ",chi_cmb)

# function for extracting the time step from a file name
def extract_steps(fn):
    split_fn = fn.split('Step')[1]
    step = np.int(split_fn.split('.asdf')[0])
    return step

# all healpix file names
hp_fns = sorted(glob.glob(simname+"LightCone*.asdf"))
n = len(hp_fns)

# these are all the time steps associated with each of the healpix files
step_fns = np.zeros(len(hp_fns),dtype=int)
for i in range(len(hp_fns)):
    step_fns[i] = extract_steps(hp_fns[i])

# comoving distance to the lensing source in Mpc
r_s = chi_cmb
# factor multiplying the standard integral (final answer should be dimensionless)
prefactor = 3*H0**2*Om_m/(2.*c**2)

'''
# load mask in nested style (same as native to abacus healpix maps)
mask_i = np.load("/mnt/store/boryanah/AbacusSummit_base_c000_ph006/lightcones/mask_nested.npy")
npix_active = np.sum(mask_i)
print(npix_active)

# percentage of sky covered
f_sky = npix_active/npix
print("f_sky [deg^2] = ",f_sky*41253)
'''

# create empty array that will save our final convergence field
kappa_i = np.zeros(npix)
# loop through all steps with light cones of interest
for step in range(step_start,step_stop+1):

    # this is because our arrays start correspond to step numbers: step_start, step_start+1, step_start+2 ... step_stop
    j = step - step_min
    step_this = steps_all[j]
    assert step_this == step, "You've messed up the counts"
    
    print("working with step = ",steps_all[j])
    
    # scale factor at the current shell
    aj = 1./(1+zs_all[j])
    # comoving distance to the current shell
    rj = chis_all[j]

    # compute width of shell
    try:
        # distance between next shell outwards and this shell
        drj = chis_all[j-1] - rj
    except:
        # distance between this shell and previous shell outwards (i.e. next shell inwards)
        drj = rj - chis_all[j+1]
    print("drj = ",drj)

    # all healpix file names which correspond to this time step
    choice_fns = np.where(step_fns == step_this)[0]
    assert (len(choice_fns) <= 3) & (len(choice_fns) > 0), "there can be at most three files in the light cones corresponding to a given step"

    # empty map
    rho_ij = np.zeros(npix)
    # loop through those files
    for choice in choice_fns:
        fn = hp_fns[choice]
        print(fn)
        f = asdf.open(fn, lazy_load=True, copy_arrays=True)
        h = f['data']['heal'][:]
        
        # get number of particles in each pixel
        unique, counts = np.unique(h,return_counts=True)
        rho_ij[unique] += counts
        f.close()
    
    # expected number of particles: delta_omega*rj**2 is the area and drj is the depth of each pixel
    dV = (delta_omega*rj**2*drj)

    # compute analytically the mean number of particles per pixel
    rho_mean = n_part*dV
    
    # the overdensity, lensing kernel and convergence
    delta_ij = rho_ij/rho_mean-1.
    lensing_kernel = ((r_s-rj)*rj/(aj*r_s))*drj
    kappa_i += delta_ij*lensing_kernel

    gc.collect()
    del rho_ij
    del delta_ij

# multiply by the prefactor
kappa_i *= prefactor

# print max and min of kappa
#print("min kappa, max kappa = ",np.min(kappa_i[mask_i]),np.max(kappa_i[mask_i]))

# save convergence map
np.save("/mnt/store/boryanah/AbacusSummit_base_c000_ph006/lightcones/kappa_nested.npy",kappa_i)
