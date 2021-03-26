import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pandas as pd

from scipy.interpolate import interp1d
from tools import *

np.random.seed(300)

# galaxy location
hod_dir = "/mnt/gosling1/boryanah/light_cone_catalog/AbacusSummit_base_c000_ph006/HOD/"
all_zs = sorted(glob.glob(os.path.join(hod_dir, "z*")))
model_no = 'model_7'

#select objects in redshift range
zrange_low = 0.45; zrange_high = 1.42

gals_fns = []
gals_zs = []
for i in range(len(all_zs)):
    z = float(all_zs[i].split('/z')[-1])

    if z < zrange_low or z > zrange_high: continue
    #if z > 0.8: continue
    gals_zs.append(z)
    gals_fns.append(os.path.join(all_zs[i], model_no, 'halos_gal_sats'))
    gals_fns.append(os.path.join(all_zs[i], model_no, 'halos_gal_cent'))


# load n(z) and f_sky
#fname='nz_blanc+abacus.txt'
fname='nz_blanc.txt'
nz = pd.read_csv(fname)


def flag(array, zrange_low = zrange_low, zrange_high = zrange_high):
    """Returns flags to clip data array into desired redshift range"""
    
    return (array >= zrange_low) & (array <= zrange_high)

#select objects in redshift range
nz = nz[flag(nz['Redshift_mid'])]
zmid = nz['Redshift_mid']
dndzddeg2 = nz['dndz/deg^2']
zmid = nz['Redshift_mid']
#abacus_area = nz['abacus_area_deg2']

# load sky coverage
dirname = '/mnt/gosling1/boryanah/light_cone_catalog/sky_coverage/'
zs_mt = np.load(dirname + "zs_mt.npy")
fs_sky = np.load(dirname + "fs_sky.npy")

#clip the file to be in the redshift range; order matters since we are using zs_mt
fs_sky = fs_sky[flag(zs_mt)]
zs_mt = zs_mt[flag(zs_mt)]

# number of galaxies per bin
dn = np.zeros(len(dndzddeg2))
dn[:] = dndzddeg2 *fs_sky * nz['Bin_width']
z_mid = np.zeros(len(dndzddeg2))
z_mid[:] = zmid[:]

n = 0
dn_down = np.zeros(len(dn))
for i in range(len(gals_zs)):
    print("pd redshift, galaxy file = ", z_mid[i], gals_zs[i])
    
    if np.abs(z_mid[i] - gals_zs[i]) > 0.01:
        print("redshifts not aligned")
        continue

    
    # load the galaxies: centrals and satellites
    gals_arr = load_gals(gals_fns[2*i: 2*(i+1)], dim=9)
    gals_pos = gals_arr[:, 0:3]
    gals_z = gals_arr[:, -3]
    N_gals = gals_pos.shape[0]

    print("mock, target number = ", N_gals, dn[i])

    factor = dn[i]/N_gals

    down = np.random.rand(N_gals) < factor
    #down = np.random.rand(N_gals) < 1.
    
    pos = gals_pos[down]
    zs = gals_z[down]
    pos = np.vstack((pos.T, zs)).T
    try:
        gals_down = np.vstack((gals_down, pos))
    except:
        gals_down = pos
    dn_down[i] = pos.shape[0]
    
plt.plot(z_mid, dn_down, label="Downsampled")
plt.plot(z_mid, dn, label="Target")
plt.legend()
plt.show()
np.save("/mnt/gosling1/boryanah/light_cone_catalog/AbacusSummit_base_c000_ph006/products/"+"gals_down_"+str(model_no)+".npy", gals_down)

