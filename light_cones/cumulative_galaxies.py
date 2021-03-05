import glob
import os

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

model_no = 'model_5'

nside = 4096

# location of the galaxy catalogs
hod_dir = "/mnt/gosling1/tkarim/light_cone_catalog/AbacusSummit_base_c000_ph006_cleaned/"
#hod_dir = "/mnt/gosling1/boryanah/light_cone_catalog/AbacusSummit_base_c000_ph006/HOD/"
save_dir = "/mnt/gosling1/boryanah/light_cone_catalog/AbacusSummit_base_c000_ph006/products/"

os.makedirs(save_dir, exist_ok=True)

want_show = 1
if want_show:
    gal_counts = np.load(os.path.join(save_dir, "gal_count_"+model_no+"_ring.npy"))
    hp.mollview(gal_counts)
    plt.show()
    quit()
    
all_zs = glob.glob(os.path.join(hod_dir, "z*"))
print(len(all_zs))

# location of the observer
origin = np.array([10., 10., 10.])


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
    cent_fns.append(os.path.join(all_zs[i], model_no, 'halos_gal_cent'))
    sats_fns.append(os.path.join(all_zs[i], model_no, 'halos_gal_sats'))
    
    gals_fns.append(os.path.join(all_zs[i], model_no, 'halos_gal_sats'))
    gals_fns.append(os.path.join(all_zs[i], model_no, 'halos_gal_cent'))
    

gals_arr = load_gals(gals_fns,dim=9)
gals_pos = gals_arr[:, 0:3]

want_show = False
if want_show:
    x_min = 0
    x_max = x_min+10.
    i = 1
    j = 2
    k = 0

    sel_gals = (gals_pos[:,k] > x_min) & (gals_pos[:,k] < x_max)
    print("number of galaxies in cross section = ",np.sum(sel_gals))

    plt.scatter(gals_pos[sel_gals,i], gals_pos[sel_gals,j], color='orangered', s=1, alpha=0.8, label='all galaxies')
    plt.legend()
    plt.axis('equal')
    plt.xlabel('Y [Mpc]')
    plt.ylabel('Z [Mpc]')
    plt.savefig("scatter.png")
    plt.show()

gals_norm = gals_pos - origin
vec_size = np.sqrt(np.sum((gals_norm)**2, axis=1))
gals_norm /= vec_size[:, None]
# sanity check
#vec_size = np.sqrt(np.sum((gals_norm)**2, axis=1))
#print(vec_size[-10:])

x = gals_norm[:, 0]
y = gals_norm[:, 1]
z = gals_norm[:, 2]

ipix_gals = hp.vec2pix(nside, x, y, z)
npix = hp.nside2npix(nside)
gal_counts = np.zeros(npix, dtype=int)
print("npix = ", npix)

ipix_un, inds, counts = np.unique(ipix_gals, return_index=True, return_counts=True)

gal_counts[ipix_un] = counts
assert np.sum(gal_counts) == len(x), "mismatch between cumulative number and galaxy total"
np.save(os.path.join(save_dir, "gal_count_"+model_no+"_ring.npy"), gal_counts)


quit()

# if want to break into centrals and satellites

cent_arr = load_gals(cent_fns,dim=9)
sats_arr = load_gals(sats_fns,dim=9)

cent_pos = cent_arr[:, 0:3]
sats_pos = sats_arr[:, 0:3]

sel_cent = (cent_pos[:,k] > x_min) & (cent_pos[:,k] < x_max)
sel_sats = (sats_pos[:,k] > x_min) & (sats_pos[:,k] < x_max)
print("number of centrals in cross section = ",np.sum(sel_cent))
print("number of satellites in cross section = ",np.sum(sel_sats))

plt.title("Cross-section of the simulation")
plt.scatter(cent_pos[sel_cent,i],cent_pos[sel_cent,j],color='dodgerblue',s=1,alpha=0.8,label='centrals')
plt.scatter(sats_pos[sel_sats,i],sats_pos[sel_sats,j],color='orangered',s=1,alpha=0.8,label='satellites')
plt.axis('equal')
plt.xlabel('Y [Mpc]')
plt.ylabel('Z [Mpc]')
plt.savefig("scatter.png")
plt.show()
