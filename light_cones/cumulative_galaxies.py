import glob
import os

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

model_no = 'model_6'
nside = 4096

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

def get_norm(gals_pos, origin):
    gals_norm = gals_pos - origin
    vec_size = np.sqrt(np.sum((gals_norm)**2, axis=1))
    gals_norm /= vec_size[:, None]
    min_dist = np.min(vec_size)
    max_dist = np.max(vec_size)
    print("min dist = ", min_dist)
    print("max dist = ", max_dist)
    return gals_norm, vec_size, min_dist, max_dist


def load_gals(fns,dim):    

    for fn in fns:
        tmp_arr = np.fromfile(fn).reshape(-1,dim)
        try:
            gal_arr = np.vstack((gal_arr,tmp_arr))
        except:
            gal_arr = tmp_arr
            
    return gal_arr


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
    
# centrals and satellites
#gals_arr = load_gals(gals_fns,dim=9)
#gals_pos = gals_arr[:, 0:3]

# og
# if want to break into centrals and satellites
cent_arr = load_gals(cent_fns,dim=9)
sats_arr = load_gals(sats_fns,dim=9)
gals_mhalo = cent_arr[:, -1]


# trying to clean the satellites that were not found (praying to the gods)
sats_zs = sats_arr[:, -3]
mask = sats_zs > 0.
sats_arr = sats_arr[mask]
print("percentage of not found satellites = ", np.sum(mask)*100/len(mask))

# and the really huge halos
sats_mhalo = sats_arr[:, -1]
mask = sats_mhalo < 1.e18
huge_inds = sats_arr[sats_mhalo > 9.e18, -2].astype(int)
#print(np.unique(huge_inds))
print(sats_arr[sats_mhalo > 9.e18][:100])
print("index of weird halo = ", np.sum(sats_mhalo > 1.e18))
sats_arr = sats_arr[mask]
print("percentage of satellites living in unphysical halos = ", np.sum(mask)*100/len(mask))

cent_pos = cent_arr[:, 0:3]
sats_pos = sats_arr[:, 0:3]

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
# TESTING with randoms
gals_pos = np.load("/mnt/gosling1/boryanah/light_cone_catalog/AbacusSummit_base_c000_ph006/products/randoms.npy")
gals_norm, vec_gals, min_gals, max_gals = get_norm(gals_pos, origin)
'''

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
    vec_size = np.hstack((vec_cent, vec_sats))
    
    bins = np.linspace(min_cent, max_cent, 1000)
    hist, bins = np.histogram(vec_size, bins=bins)

    bin_cents = (bins[1:] + bins[:-1])*0.5
    plt.plot(bin_cents, hist/bin_cents**2)
    plt.show()

    np.save("data/bin_cents.npy", bin_cents)
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
