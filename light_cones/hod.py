import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import asdf

from tools import *

# light cone directory
#sandy = 1; buba = 0 # ph000 same HOD
sandy = 0; buba = 1 # this is snapshots I produced but if I remember correctly, they have small-scale bs
#sandy = 0; buba = 0 # light cone catalog ph006 (kinda broken)

if sandy:
    # sandy
    light_dir = "/mnt/marvin1/boryanah/scratch/data_mocks_summit_new/AbacusSummit_base_c000_ph000/"
    hod_dir = light_dir
elif buba:
    light_dir = "/mnt/gosling1/boryanah/AbacusHOD_scratch/mocks/AbacusSummit_base_c000_ph006"
    hod_dir = light_dir
else:
    light_dir = "/mnt/gosling1/boryanah/light_cone_catalog/AbacusSummit_base_c000_ph006/"
    # galaxy location
    hod_dir = light_dir+"HOD/"
    
all_zs = sorted(glob.glob(os.path.join(hod_dir, "z*")))
model_no = 'model_7'
#model_no = 'model_6'
#model_no = 'model_5'
#model_no = 'model_5_rsd'
#model_no = 'model_4_rsd'
#model_no = 'model_3_rsd'
#model_no = 'model_2_rsd'

# halo location
halo_dir = light_dir+"halos_light_cones/"
halo_fns = sorted(glob.glob(os.path.join(halo_dir, "z*")))

# min and max redshift
if buba:
    #z_min = 0.75
    #z_max = 0.85
    # TESTING we now have this after the cleaning
    z_min = 0.46
    z_max = 0.55
elif sandy:
    z_min = 0.46
    z_max = 0.55
else:
    #z_min = 1.543#0.75
    #z_max = 1.563#0.85
    z_min = 0.75
    z_max = 0.85
    
gals_fns = []
sats_fns = []
cent_fns = []
for i in range(len(all_zs)):
    z = float(all_zs[i].split('/z')[-1])

    if z > z_max or z < z_min: continue
    print(z)
    if sandy:
        # sandy
        #gals_fns.append(os.path.join(all_zs[i], 'galaxies_rsd', 'ELGs.dat'))
        gals_fns.append(os.path.join(all_zs[i], 'galaxies', 'ELGs.dat'))
        
    elif buba:
        # buba
        gals_fns.append(os.path.join(all_zs[i], 'galaxies', 'ELGs.dat'))
    else:
        gals_fns.append(os.path.join(all_zs[i], model_no, 'halos_gal_sats'))
        gals_fns.append(os.path.join(all_zs[i], model_no, 'halos_gal_cent'))
        sats_fns.append(os.path.join(all_zs[i], model_no, 'halos_gal_sats'))
        cent_fns.append(os.path.join(all_zs[i], model_no, 'halos_gal_cent'))

print(gals_fns)

info_fns = []
for i in range(len(halo_fns)):
    z = float(halo_fns[i].split('/z')[-1])

    if z > z_max or z < z_min: continue

    info_fns.append(os.path.join(halo_fns[i], 'halo_info_lc.asdf'))

# load the galaxies: centrals and satellites
if sandy or buba:
    from astropy.io import ascii
    gals_arr = ascii.read(gals_fns[0])  
    #gals_arr = np.loadtxt(gals_fns[0])
    haloid = gals_arr['id']
    gals_mhalo = gals_arr['mass']
    pos = np.vstack((gals_arr['x'], gals_arr['y'], gals_arr['z'])).T
else:
    cent_arr = load_gals(cent_fns, dim=9)
    cent_zs = cent_arr[:, -3]
    cent_haloid = cent_arr[:, -2].astype(int)
    cent_mhalo = cent_arr[:, -1]
    cent_pos = cent_arr[:, :3]

    sats_arr = load_gals(sats_fns, dim=9)
    sats_zs = sats_arr[:, -3]
    sats_haloid = sats_arr[:, -2].astype(int)
    sats_mhalo = sats_arr[:, -1]
    sats_pos = sats_arr[:, :3]
    
    haloid = np.hstack((cent_haloid, sats_haloid))
    pos = np.vstack((sats_pos, cent_pos))
    gals_mhalo = np.hstack((cent_mhalo, sats_mhalo))

    print("number of centrals = ", len(cent_mhalo))
    print("number of satellites = ", len(sats_mhalo))
    
i_sort = np.argsort(haloid)

haloid = haloid[i_sort]
pos = pos[i_sort]
gals_mhalo = gals_mhalo[i_sort]


id, inds, counts = np.unique(haloid, return_index=True, return_counts=True)

Lbox = 2000.
origin = np.array([10., 10., 10.])
if sandy or buba:
    origin -= Lbox/2.
gals_norm, vec_size, min_dist, max_dist = get_norm(pos, origin)
# TESTING the min and max are wrong for the old models cause of wrapping
mid_dist = 0.5*(min_dist+max_dist)
print("min dist = ", min_dist)
print("max dist = ", max_dist)
print("mid dist = ", mid_dist)

hpos = pos[inds]
hpos_each = np.repeat(hpos, counts, axis=0)
pos_diff = np.sqrt(np.sum((pos-hpos_each)**2, axis=1))

arg = np.argmax(counts)

i_sort = np.argsort(counts)[::-1]
print("max gals = ", counts[i_sort][:50])

print("most galaxies = ", counts[arg])
print("all gal pos of the halo with most galaxies = ", pos[id[arg] == haloid])
print("mass of the halo with most galaxies = ", gals_mhalo[id[arg] == haloid])
print("id of the halo with most galaxies = ", haloid[id[arg] == haloid])
print("index of the galaxies in the halo with most galaxies = ", np.where(id[arg] == haloid))

print("most massive halo = %.3e"%np.max(gals_mhalo))
print("least massive halo = %.3e"%np.min(gals_mhalo))

print("satellite fraction from distance to center = ", np.sum(pos_diff>0.)*100./len(pos_diff))
pos_diff = pos_diff[pos_diff > 0]


print("min diff = ", np.min(pos_diff))
print("max diff = ", np.max(pos_diff))
print("objects further than 20 mpc/h = ", np.sum(pos_diff > 20)*100/len(pos_diff))

bins = np.linspace(np.min(pos_diff)-1., np.max(pos_diff)+1., 2001)
#bins = np.linspace(np.min(pos_diff)-1., np.min(pos_diff)+10., 2001)
hist, bins = np.histogram(pos_diff, bins=bins)
binc = (bins[1:] + bins[:-1])*.5

plt.plot(binc, hist)
plt.show()


    
# load the halos

if sandy:
    # obviously bullshit
    #f = asdf.open("data_halo/halo_info_lc.asdf", copy_arrays=True, lazy_load=True)
    f = asdf.open("/mnt/gosling2/bigsims/AbacusSummit_base_c000_ph000/halos/z0.500/halo_info/halo_info_000.asdf", copy_arrays=True, lazy_load=True)
elif buba:
    f = asdf.open("/mnt/gosling1/boryanah/light_cone_catalog/AbacusSummit_base_c000_ph006/halos_light_cones/z0.500/lc_halo_info.asdf", copy_arrays=True, lazy_load=True)
    header = asdf.open("/mnt/gosling1/boryanah/light_cone_catalog/AbacusSummit_base_c000_ph006/halos_light_cones/z0.500/lc_pid_rv.asdf", copy_arrays=True, lazy_load=True)['header']
    f['header'] = header
else:
    f = asdf.open("data_halo/halo_info_lc.asdf", copy_arrays=True, lazy_load=True)
    #f = asdf.open(info_fns[0])#, copy_arrays=True, lazy_load=True)
# halo masses
M = f['data']['N'].astype(float)
m_part = f['header']['ParticleMassHMsun']
M *= m_part
f.close()

#np.unique(gals_haloid, return_counts=True)
m_bins = np.logspace(11, 15, 31)

# compute the cumulative HOD
if sandy or buba:
    hist, bins = np.histogram(gals_mhalo, bins=m_bins)
else:
    hist_cent, bins = np.histogram(cent_mhalo, bins=m_bins)
    hist_sats, bins = np.histogram(sats_mhalo, bins=m_bins)
hist_norm, bins = np.histogram(M, bins=m_bins)

if sandy:
    hist_norm *= 34


m_binc = (m_bins[1:]+m_bins[:-1])*.5

plt.plot(m_binc, hist_norm)
plt.xscale('log')
plt.yscale('log')
plt.show()

#plt.plot(m_binc, hist)
if sandy:
    plt.plot(m_binc, hist/hist_norm)
    np.save("data/hist_sandy.npy", hist)
    np.save("data/hist_norm_sandy.npy", hist_norm)
elif buba:
    plt.plot(m_binc, hist/hist_norm)
    np.save("data/hist_buba.npy", hist)
    np.save("data/hist_norm_buba.npy", hist_norm)
else:
    np.save("data/hist_cent.npy", hist_cent)
    np.save("data/hist_sats.npy", hist_sats)
    np.save("data/hist_norm.npy", hist_norm)
    plt.plot(m_binc, hist_cent/hist_norm, 'k--')
    plt.plot(m_binc, hist_sats/hist_norm, 'k-')
    plt.ylim([1.e-7, 10.])
np.save("data/m_binc.npy", m_binc)
plt.xscale('log')
plt.yscale('log')
plt.show()
