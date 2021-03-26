import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from scipy.interpolate import interp1d
import Corrfunc
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from Corrfunc.utils import convert_rp_pi_counts_to_wp

from tools import *

want_show = False
if want_show:
    wp = np.load("data/wp.npy")
    bin_cents = np.load("data/rbin_cents.npy")
    plt.plot(bin_cents, wp*bin_cents)
    plt.xscale('log')
    plt.show()
    quit()

# all redshifts, steps and comoving distances of light cones files; high z to low z
zs_all = np.load("data_headers/redshifts.npy")
chis_all = np.load("data_headers/coord_dist.npy")
zs_all[-1] = np.float('%.1f'%zs_all[-1])

# get functions relating chi and z
chi_of_z = interp1d(zs_all,chis_all)
z_of_chi = interp1d(chis_all,zs_all)

# galaxy location
hod_dir = "/mnt/gosling1/boryanah/light_cone_catalog/AbacusSummit_base_c000_ph006/HOD/"
all_zs = sorted(glob.glob(os.path.join(hod_dir, "z*")))
#model_no = 'model_6'
model_no = 'model_7'

# min and max redshift
z_min = 0.475#0.4625
z_max = 0.5375

gals_fns = []
for i in range(len(all_zs)):
    z = float(all_zs[i].split('/z')[-1])

    if z > z_max or z < z_min: continue
    #if z > 0.8: continue
    print(z)    
    gals_fns.append(os.path.join(all_zs[i], model_no, 'halos_gal_sats'))
    gals_fns.append(os.path.join(all_zs[i], model_no, 'halos_gal_cent'))

downsampled = False
if downsampled:
    # downsampled NEW
    gals_arr = np.load("/mnt/gosling1/boryanah/light_cone_catalog/AbacusSummit_base_c000_ph006/products/gals_down_"+str(model_no)+".npy")
    gals_zs = gals_arr[:, -1]
else:
    # load the galaxies: centrals and satellites
    gals_arr = load_gals(gals_fns,dim=9)
    gals_zs = gals_arr[:, -3]

gals_pos = gals_arr[:, :3]
# shouldn't really matter....
#mask = (gals_zs < z_max) & (gals_zs > z_min)

# load randoms
rands_arr = np.load("/mnt/gosling1/boryanah/light_cone_catalog/AbacusSummit_base_c000_ph006/products/randoms.npy")
rands_pos = rands_arr[:, :3]
print("number of randoms = ", rands_pos.shape[0])

# origin location
origin = np.array([10., 10., 10.])

# get the unit vectors and comoving distances to the observer
gals_norm, gals_chis, min_gals, max_gals = get_norm(gals_pos, origin)
# the randoms are centered at zero (maybe change in generate randoms)
rands_norm, rands_chis, min_rands, max_rands = get_norm(rands_pos, origin)


# where we're gonna define the beginning and ending
max_gals = chi_of_z(z_max)
min_gals = chi_of_z(z_min)
#min_gals = chi_of_z(0.51)
#2,2,0.4625,0.5,0.07499999999999996,125.78086614725916,5273.252466444546
#3,3,0.5375,0.575,0.07499999999999996,53.90608549168249,5261.3187599267
gal_num = 5273.252466444546*0.07499999999999996*125.78086614725916
print("target number = ", gal_num)
print("min, max gals = ", min_gals, max_gals)


# cut the galaxies
mask = (gals_chis < max_gals) & (gals_chis > min_gals)
gals_norm = gals_norm[mask]
gals_chis = gals_chis[mask]

print("number of true galaxies", np.sum(mask))

want_scatter = 1
if want_scatter:
    x = gals_pos[mask, 0]
    y = gals_pos[mask, 1]
    z = gals_pos[mask, 2]
    choice = (x > 300) & (x < 350)
    plt.scatter(y[choice], z[choice], s=0.1)
    plt.axis('equal')
    #plt.show()


# cut the randoms (cause we generated only one random catalog)
mask = (rands_chis < max_gals) & (rands_chis > min_gals)
rands_norm = rands_norm[mask]
rands_chis = rands_chis[mask]

want_scatter = 1
if want_scatter:
    x = rands_pos[mask, 0]
    y = rands_pos[mask, 1]
    z = rands_pos[mask, 2]
    choice = (x > 300) & (x < 350)
    plt.scatter(y[choice], z[choice], s=0.1)
    plt.axis('equal')
    plt.show()

# convert the unit vectors into RA and DEC
RA, DEC, CZ = get_ra_dec_chi(gals_norm, gals_chis)
RAND_RA, RAND_DEC, RAND_CZ = get_ra_dec_chi(rands_norm, rands_chis)

want_hist = 0
if want_hist:
    print("min, max RA = ", np.min(RA), np.max(RA))
    print("min, max DEC = ", np.min(DEC), np.max(DEC))
    print("min, max CZ = ", np.min(CZ), np.max(CZ))

    print("min, max RAND_RA = ", np.min(RAND_RA), np.max(RAND_RA))
    print("min, max RAND_DEC = ", np.min(RAND_DEC), np.max(RAND_DEC))
    print("min, max RAND_CZ = ", np.min(RAND_CZ), np.max(RAND_CZ))

    bins = np.linspace(np.min(CZ), np.max(CZ), 2001)
    binc = (bins[1:] + bins[:-1])*0.5
    hist_cz, edges = np.histogram(CZ, bins=bins, density=True)
    hist_rcz, edges = np.histogram(RAND_CZ, bins=bins, density=True)

    plt.title("CZ")
    plt.plot(binc, hist_cz, label='CZ')
    plt.plot(binc, hist_rcz, label='RAND_CZ')
    plt.legend()
    plt.show()

    bins = np.linspace(-10., 370., 2001)
    binc = (bins[1:] + bins[:-1])*0.5
    hist_ra, edges = np.histogram(RAND_RA, bins=bins)
    hist_dec, edges = np.histogram(RAND_DEC, bins=bins)

    plt.title("RA")
    plt.plot(binc, hist_ra)
    plt.show()

    plt.title("DEC")
    plt.plot(binc, hist_dec)
    plt.show()
    quit()

N = len(RA)
RAND_N = len(RAND_RA)
print("number of galaxies = ", N)
print("number of randoms = ", RAND_N)

# Number of threads to use
nthreads = 16

# Specify cosmology (1->LasDamas, 2->Planck)
# doesn't matter if giving chi
cosmology = 1

# Create the bins array
rmin = 0.2# 0.1
rmax = 31.#20.0
nbins = 8#20
rbins = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
bin_cents = 0.5*(rbins[1:]+rbins[:-1])

# Specify the distance to integrate along line of sight
pimax = 31#40.0

# Specify that an autocorrelation is wanted
autocorr = 1
DD_counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, rbins, RA, DEC, CZ, is_comoving_dist=True)

# Cross pair counts in DR
autocorr = 0
DR_counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, rbins, RA, DEC, CZ, RA2=RAND_RA, DEC2=RAND_DEC, CZ2=RAND_CZ, is_comoving_dist=True)

# Auto pairs counts in RR
autocorr = 1
RR_counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, rbins, RAND_RA, RAND_DEC, RAND_CZ, is_comoving_dist=True)

# All the pair counts are done, get the angular correlation function
wp = convert_rp_pi_counts_to_wp(N, N, RAND_N, RAND_N, DD_counts, DR_counts, DR_counts, RR_counts, nbins, pimax)

if downsampled:
    np.save("data/wp_down.npy", wp)
else:
    np.save("data/wp.npy", wp)
np.save("data/rbin_cents.npy", bin_cents)

plt.plot(bin_cents, wp*bin_cents)
plt.xscale('log')
plt.show()
