import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from scipy.interpolate import interp1d
import Corrfunc
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from Corrfunc.utils import convert_rp_pi_counts_to_wp

def load_gals(fns,dim):    

    for fn in fns:
        tmp_arr = np.fromfile(fn).reshape(-1,dim)
        try:
            gal_arr = np.vstack((gal_arr,tmp_arr))
        except:
            gal_arr = tmp_arr
            
    return gal_arr

want_show = True
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
model_no = 'model_6'

gals_fns = []
for i in range(len(all_zs)):
    z = float(all_zs[i].split('/z')[-1])

    if z > 1.175 or z < 1.03: continue
    #if z > 0.8: continue
    print(z)    
    gals_fns.append(os.path.join(all_zs[i], model_no, 'halos_gal_sats'))
    gals_fns.append(os.path.join(all_zs[i], model_no, 'halos_gal_cent'))

# load the galaxies: centrals and satellites
gals_arr = load_gals(gals_fns,dim=9)
gals_zs = gals_arr[:, -3]
mask = gals_zs > 0.
gals_pos = gals_arr[mask, 0:3]
gals_zs = gals_zs[mask]
gals_chis = chi_of_z(gals_zs)
origin = np.array([10., 10., 10.])
N_gals = len(gals_chis)

print("number of galaxies = ", N_gals)
print("chis = ", gals_chis[:10])

# load randoms
rands_pos = np.load("data/randoms.npy")
print("number of randoms = ", rands_pos.shape[0])

def get_norm(gals_pos, origin):
    gals_norm = gals_pos - origin
    vec_size = np.sqrt(np.sum((gals_norm)**2, axis=1))
    gals_norm /= vec_size[:, None]
    min_dist = np.min(vec_size)
    max_dist = np.max(vec_size)
    print("min dist = ", min_dist)
    print("max dist = ", max_dist)
    return gals_norm, vec_size, min_dist, max_dist


def get_ra_dec_chi(norm, chis):
    theta, phi = hp.vec2ang(norm)
    ra = phi
    dec = np.pi/2. - theta
    ra *= 180./np.pi
    dec *= 180./np.pi
    print("vecs = ", chis[:10])
    print("max ra = ", np.max(ra))
    print("min ra = ", np.min(ra))
    print("max dec = ", np.max(dec))
    print("min dec = ", np.min(dec))
    
    return ra, dec, chis

# get the unit vectors and comoving distances to the observer
gals_norm, gals_chis, min_gals, max_gals = get_norm(gals_pos, origin)
# the randoms are centered at zero (maybe change in generate randoms)
rands_norm, rands_chis, min_rands, max_rands = get_norm(rands_pos, np.array([0, 0, 0]))

# cut the randoms (cause we generated only one random catalog)
mask = (rands_chis < max_gals) & (rands_chis > min_gals)
rands_norm = rands_norm[mask]
rands_chis = rands_chis[mask]

# convert the unit vectors into RA and DEC
RA, DEC, CZ = get_ra_dec_chi(gals_norm, gals_chis)
RAND_RA, RAND_DEC, RAND_CZ = get_ra_dec_chi(rands_norm, rands_chis)
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
rmin = 0.1
rmax = 20.0
nbins = 20
rbins = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
bin_cents = 0.5*(rbins[1:]+rbins[:-1])

# Specify the distance to integrate along line of sight
pimax = 40.0

# Specify that an autocorrelation is wanted
autocorr = 1
DD_counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, rbins, RA, DEC, CZ, is_comoving_dist=True)

# Cross pair counts in DR
autocorr=0
DR_counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, rbins, RA, DEC, CZ, RA2=RAND_RA, DEC2=RAND_DEC, CZ2=RAND_CZ, is_comoving_dist=True)

# Auto pairs counts in RR
autocorr=1
RR_counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, rbins, RAND_RA, RAND_DEC, RAND_CZ, is_comoving_dist=True)

# All the pair counts are done, get the angular correlation function
wp = convert_rp_pi_counts_to_wp(N, N, RAND_N, RAND_N, DD_counts, DR_counts, DR_counts, RR_counts, nbins, pimax)


np.save("data/wp.npy", wp)
np.save("data/rbin_cents.npy", bin_cents)

plt.plot(bin_cents, wp*bin_cents)
plt.xscale('log')
plt.show()
