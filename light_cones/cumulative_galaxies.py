import glob
import os

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

from scipy.interpolate import interp1d

from tools import *

model_no = 'model_7'
nside = 4096


# all redshifts, steps and comoving distances of light cones files; high z to low z
zs_all = np.load("data_headers/redshifts.npy")
chis_all = np.load("data_headers/coord_dist.npy")
zs_all[-1] = np.float('%.1f'%zs_all[-1])

# get functions relating chi and z
chi_of_z = interp1d(zs_all,chis_all)
z_of_chi = interp1d(chis_all,zs_all)

# location of the galaxy catalogs
#hod_dir = "/mnt/gosling1/tkarim/light_cone_catalog/AbacusSummit_base_c000_ph006_cleaned/"
hod_dir = "/mnt/gosling1/boryanah/light_cone_catalog/AbacusSummit_base_c000_ph006/HOD/"
save_dir = "/mnt/gosling1/boryanah/light_cone_catalog/AbacusSummit_base_c000_ph006/products/"

os.makedirs(save_dir, exist_ok=True)

want_show = 0
if want_show:
    gal_counts = np.load(os.path.join(save_dir, "gal_count_"+model_no+"_ring.npy"))
    nonzero = np.sum(gal_counts > 0)
    gal_avg = np.sum(gal_counts)/nonzero
    print(gal_avg)
    gal_counts[gal_counts == 0] = 1.e-6
    gal_counts = np.log10(gal_counts/gal_avg+1.e-3)
    hp.mollview(gal_counts, cmap='inferno')
    plt.show()
    quit()
    
all_zs = sorted(glob.glob(os.path.join(hod_dir, "z*")))
print(len(all_zs))

# location of the observer
origin = np.array([10., 10., 10.])

cent_fns = []
sats_fns = []
gals_fns = []
for i in range(len(all_zs)):
    z = float(all_zs[i].split('/z')[-1])

    #if z > 1.6 or z < 0.8: continue
    #if z > 0.8: continue
    #print(z)
    cent_fns.append(os.path.join(all_zs[i], model_no, 'halos_gal_cent'))
    sats_fns.append(os.path.join(all_zs[i], model_no, 'halos_gal_sats'))
    
    gals_fns.append(os.path.join(all_zs[i], model_no, 'halos_gal_sats'))
    gals_fns.append(os.path.join(all_zs[i], model_no, 'halos_gal_cent'))


# START COMMENT if want radoms or downsampled
'''
# centrals and satellites
#gals_arr = load_gals(gals_fns,dim=9)
#gals_pos = gals_arr[:, 0:3]

# if want to break into centrals and satellites
cent_arr = load_gals(cent_fns,dim=9)
sats_arr = load_gals(sats_fns,dim=9)
gals_mhalo = cent_arr[:, -1]

# centrals and satellites position
cent_pos = cent_arr[:, 0:3]
sats_pos = sats_arr[:, 0:3]

# compute the distance to the observer and unit vector
cent_norm, vec_cent, min_cent, max_cent = get_norm(cent_pos, origin)
sats_norm, vec_sats, min_sats, max_sats = get_norm(sats_pos, origin)
print("max_cent/sats = ", max_cent, max_sats)
print("min_cent/sats = ", min_cent, min_sats)

# cut the last bitsie of centrals
cent_norm = cent_norm[vec_cent < max_sats]
cent_pos = cent_pos[vec_cent < max_sats]

# stacking the normalized centrals and satellites
gals_norm = np.vstack((cent_norm, sats_norm))
'''
# END COMMENT if want radoms or downsampled

# randoms or downsampled
randoms = 0
if randoms:
    gals_arr = np.load("/mnt/gosling1/boryanah/light_cone_catalog/AbacusSummit_base_c000_ph006/products/randoms.npy")
else:
    gals_arr = np.load("/mnt/gosling1/boryanah/light_cone_catalog/AbacusSummit_base_c000_ph006/products/gals_down_"+model_no+".npy")
gals_pos = gals_arr[:, :3]
gals_zs = gals_arr[:, -1]
gals_norm, vec_gals, min_gals, max_gals = get_norm(gals_pos, origin)

# if want chi instead of distance (particles take the light cone shell values)
#vec_gals = chi_of_z(gals_zs)

want_show = 0
if want_show:
    x_min = 0
    x_max = x_min+10.
    i = 1
    j = 2
    k = 0

    sel_gals = (gals_pos[:,k] > x_min) & (gals_pos[:,k] < x_max)
    print("number of galaxies in cross section = ",np.sum(sel_gals))
    # og
    '''
    sel_cent = (cent_pos[:,k] > x_min) & (cent_pos[:,k] < x_max)
    sel_sats = (sats_pos[:,k] > x_min) & (sats_pos[:,k] < x_max)
    print("number of centrals in cross section = ",np.sum(sel_cent))
    print("number of satellites in cross section = ",np.sum(sel_sats))
    
    plt.scatter(cent_pos[sel_cent,i],cent_pos[sel_cent,j],color='dodgerblue',s=1,alpha=0.8,label='centrals')
    plt.scatter(sats_pos[sel_sats,i],sats_pos[sel_sats,j],color='orangered',s=1,alpha=0.8,label='satellites')
    '''
    plt.scatter(gals_pos[sel_gals,i], gals_pos[sel_gals,j], color='orangered', s=1, alpha=0.8, label='all galaxies')
    plt.legend()
    plt.axis('equal')
    plt.xlabel('Y [Mpc]')
    plt.ylabel('Z [Mpc]')
    plt.savefig("scatter.png")
    plt.show()


want_show = 0
if want_show:
    #vec_gals = np.hstack((vec_cent, vec_sats))
    #min_gals = 1260
    #max_gals = 1450
    num = 2001
    
    bins = np.linspace(min_gals, max_gals, num)
    hist, bins = np.histogram(vec_gals, bins=bins)

    bin_cents = (bins[1:] + bins[:-1])*0.5
    plt.plot(bin_cents, hist/bin_cents**2)
    plt.show()

    np.save("data/bin_cents.npy", bin_cents)
    if randoms:
        np.save("data/hist_rand.npy", hist/bin_cents**2)
    else:
        np.save("data/hist.npy", hist/bin_cents**2)
    quit()
    
x = gals_norm[:, 0]
y = gals_norm[:, 1]
z = gals_norm[:, 2]

ipix_gals = hp.vec2pix(nside, x, y, z)
npix = hp.nside2npix(nside)
gal_counts = np.zeros(npix, dtype=int)
print("npix = ", npix)

ipix_un, inds, counts = np.unique(ipix_gals, return_index=True, return_counts=True)

gal_counts[ipix_un] = counts
print("total number of galaxies = ", len(x))
assert np.sum(gal_counts) == len(x), "mismatch between cumulative number and galaxy total"
np.save(os.path.join(save_dir, "gal_count_"+model_no+"_ring.npy"), gal_counts)

want_show = 1
if want_show:
    #nonzero = np.sum(gal_counts > 0)
    #gal_avg = np.sum(gal_counts)/nonzero
    gal_avg = 1.
    
    gal_counts[gal_counts == 0] = 1.e-6
    gal_counts = np.log10(gal_counts/gal_avg+1.e-3)
    hp.mollview(gal_counts, cmap='inferno')
    plt.show()
