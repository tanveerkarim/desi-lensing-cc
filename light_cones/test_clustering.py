import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from scipy.interpolate import interp1d
import Corrfunc
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from Corrfunc.utils import convert_rp_pi_counts_to_wp
from astropy.io import ascii

from tools import *
from generate_randoms import gen_rand


# size of sim box
Lbox = 2000.
sandy = 0; buba = 1
#sandy = 1; buba = 0

# all redshifts, steps and comoving distances of light cones files; high z to low z
zs_all = np.load("data_headers/redshifts.npy")
chis_all = np.load("data_headers/coord_dist.npy")
zs_all[-1] = np.float('%.1f'%zs_all[-1])

# get functions relating chi and z
chi_of_z = interp1d(zs_all,chis_all)
z_of_chi = interp1d(chis_all,zs_all)

# galaxy location
if sandy:
    hod_dir = "/mnt/marvin1/boryanah/scratch/data_mocks_summit_new/AbacusSummit_base_c000_ph000/"
if buba:
    hod_dir = "/mnt/gosling1/boryanah/AbacusHOD_scratch/mocks/AbacusSummit_base_c000_ph006"
all_zs = sorted(glob.glob(os.path.join(hod_dir, "z*")))

# min and max redshift
if sandy:
    z_min = 0.45#0.4#0.45
    z_max = 0.55#0.7#0.55
if buba:
    #z_min = 0.7625
    #z_max = 0.8375
    # TESTING we now have this after the cleaning!!!
    z_min = 0.475
    z_max = 0.5375
    
# where we're gonna define the beginning and ending
max_gals = chi_of_z(z_max)
min_gals = chi_of_z(z_min)
#2,2,0.4625,0.5,0.07499999999999996,125.78086614725916,5273.252466444546
#3,3,0.5375,0.575,0.07499999999999996,53.90608549168249,5261.3187599267
if sandy:
    #gal_num = 5273.252466444546*0.07499999999999996*125.78086614725916
    # TESTING!!!!!
    gal_num = 5273.252466444546*0.07499999999999996*800.78086614725916
if buba:
    # TESTING we now have this after the cleaning
    # z = 0.5
    #gal_num = 5273.252466444546*0.07499999999999996*125.78086614725916
    # z = 0.8
    #gal_num = 5273.252466444546*0.07499999999999996*2000.
    gal_num = 5273.252466444546*0.07499999999999996*500.
print("target number = ", gal_num)
print("min, max gals = ", min_gals, max_gals)

    
gals_fns = []
for i in range(len(all_zs)):
    z = float(all_zs[i].split('/z')[-1])

    if z > z_max or z < z_min: continue
    print("z = ", z)

    if sandy:
        #gals_fns.append(os.path.join(all_zs[i], 'galaxies_rsd', 'ELGs.dat'))
        gals_fns.append(os.path.join(all_zs[i], 'galaxies', 'ELGs.dat'))
    if buba:
        gals_fns.append(os.path.join(all_zs[i], 'galaxies', 'ELGs.dat'))


# sandy or buba
gals_arr = ascii.read(gals_fns[0])  
gals_pos = np.vstack((gals_arr['x'], gals_arr['y'], gals_arr['z'])).T
#np.save("data_halo/gals_pos.npy", gals_pos)
#gals_pos = np.load("data_halo/gals_pos.npy")
#box_offset = Lbox/2.
box_offset = 0.
gals_pos += box_offset

# location of the observer
origin = np.array([-990., -990., -990.])
# TESTING!!!!!!!
#origin = np.array([-2000., -2000., -2000.])# works!
#origin = np.array([-3000., -3000., -3000.])
#origin = np.array([-2500., -2500., -2500.])
#origin = np.array([-3500., -3500., -3500.])
corner = np.array([1., 1, 1])*(-Lbox/2.) + box_offset
#corner = np.array([0., 0, 1])*(-Lbox/2.) + box_offset
# TESTING!!!!!!
N_gals = gals_pos.shape[0]
print("number of galaxies = ", N_gals)

# get the unit vectors and comoving distances to the observer
gals_norm, gals_chis, gals_min, gals_max = get_norm(gals_pos, origin+box_offset)

if buba:
    min_gals = gals_min
    max_gals = gals_max
if sandy:
    # TESTING !!!!!!!!
    min_gals = np.sqrt(np.sum(((origin+box_offset)-corner)**2))+1000
    max_gals = min_gals+200.
    
# cut the galaxies
mask_gals = (gals_chis < max_gals) & (gals_chis > min_gals)
gals_norm = gals_norm[mask_gals]
gals_chis = gals_chis[mask_gals]
gals_pos = gals_pos[mask_gals]
N_gals = np.sum(mask_gals)

# downsample galaxies
down = gal_num/len(gals_chis)
print("down factor = ", down)
mask_gals = np.random.rand(len(gals_chis)) < down
gals_norm = gals_norm[mask_gals]
gals_chis = gals_chis[mask_gals]
#gals_pos = gals_pos[mask_gals]
N_gals = np.sum(mask_gals)

print("target number = ", gal_num)
print("true number = ", len(gals_chis))

generate_randoms = True
if generate_randoms:
    if buba:
        rands_pos = gen_rand(N_gals, gals_chis, fac=120, remove_edges=True, origin=origin, box_offset=False)[:, :3]
    else:
        rands_pos = gen_rand(N_gals, gals_chis, fac=120, origin=origin, box_offset=False)[:, :3]
    #np.save("data_halo/rands_pos.npy", rands_pos)
    #rands_pos = np.load("data_halo/rands_pos.npy")
else:
    # load randoms
    rands_arr = np.load("/mnt/gosling1/boryanah/light_cone_catalog/AbacusSummit_base_c000_ph006/products/randoms.npy")
    rands_pos = rands_arr[:, :3]
    print("number of randoms = ", rands_pos.shape[0])

# the randoms are centered at zero (maybe change in generate randoms) (not anymore)
rands_norm, rands_chis, min_rands, max_rands = get_norm(rands_pos, origin+box_offset)

# cut the randoms (cause we generated only one random catalog)
mask_rands = (rands_chis < max_gals) & (rands_chis > min_gals)
rands_norm = rands_norm[mask_rands]
rands_chis = rands_chis[mask_rands]


want_scatter = 1
if want_scatter:
    x = rands_pos[mask_rands, 0]
    y = rands_pos[mask_rands, 1]
    z = rands_pos[mask_rands, 2]
    choice = (x > -700+box_offset) & (x < -650+box_offset)
    plt.scatter(y[choice], z[choice], s=0.1)
    plt.axis('equal')
    plt.show()

want_scatter = 1
if want_scatter:
    x = gals_pos[mask_gals, 0]
    y = gals_pos[mask_gals, 1]
    z = gals_pos[mask_gals, 2]
    choice = (x > -700+box_offset) & (x < -650+box_offset)
    plt.scatter(y[choice], z[choice], s=0.1)
    plt.axis('equal')
    plt.show()


# convert the unit vectors into RA and DEC
RA, DEC, CZ = get_ra_dec_chi(gals_norm, gals_chis)
RAND_RA, RAND_DEC, RAND_CZ = get_ra_dec_chi(rands_norm, rands_chis)

want_hist = 1
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
    plt.plot(binc, hist_cz, label="true")
    plt.plot(binc, hist_rcz, label="random")
    plt.legend()
    plt.show()

    bins = np.linspace(-10., 370., 2001)
    binc = (bins[1:] + bins[:-1])*0.5
    hist_ra, edges = np.histogram(RA, bins=bins, density=True)
    hist_rra, edges = np.histogram(RAND_RA, bins=bins, density=True)
    hist_dec, edges = np.histogram(DEC, bins=bins, density=True)
    hist_rdec, edges = np.histogram(RAND_DEC, bins=bins, density=True)

    plt.title("RA")
    plt.plot(binc, hist_ra, label="true")
    plt.plot(binc, hist_rra, label="random")
    plt.legend()
    plt.show()

    plt.title("DEC")
    plt.plot(binc, hist_dec, label="true")
    plt.plot(binc, hist_rdec, label="random")
    plt.legend()
    plt.show()

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
pimax = 30.#40.0

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

if sandy:
    np.save("data/wp_sandy.npy", wp)
if buba:
    np.save("data/wp_buba.npy", wp)
np.save("data/rbin_cents.npy", bin_cents)

wp_s = np.load("/home/boryanah/repos/abacusutils/scripts/hod/data/wp.npy")
bc_s = np.load("/home/boryanah/repos/abacusutils/scripts/hod/data/bin_cents.npy")
plt.plot(bc_s, wp_s*bc_s, label='full snapshot')

plt.plot(bin_cents, wp*bin_cents)
plt.legend()
plt.xscale('log')
plt.show()
