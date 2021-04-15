import os

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def get_vertices_cube(units=0.5,N=3):
    vertices = 2*((np.arange(2**N)[:,None] & (1 << np.arange(N))) > 0) - 1
    return vertices*units

def is_in_cube(x_pos,y_pos,z_pos,verts):
    x_min = np.min(verts[:,0])
    x_max = np.max(verts[:,0])
    y_min = np.min(verts[:,1])
    y_max = np.max(verts[:,1])
    z_min = np.min(verts[:,2])
    z_max = np.max(verts[:,2])

    mask = (x_pos > x_min) & (x_pos <= x_max) & (y_pos > y_min) & (y_pos <= y_max) & (z_pos > z_min) & (z_pos <= z_max)

    #print("f_sky [deg^2] = ",np.sum(mask)/len(x_pos)*41253.)
    
    return mask

os.makedirs("/mnt/gosling1/boryanah/light_cone_catalog/sky_coverage/", exist_ok=True)

# simulation constants
nside = 1024#256
npix = hp.nside2npix(nside)
ipix = np.arange(npix)
x_cart, y_cart, z_cart = hp.pix2vec(nside, ipix)
Lbox = 2000. # Mpc/h
full_sky = 360.**2/np.pi # deg^2
origin = np.array([-990.,-990.,-990.])
sim_name = "AbacusSummit_base_c000_ph006"

# all redshifts, steps and comoving distances of light cones files; high z to low z
zs_all = np.load(os.path.expanduser("~/repos/abacus_lc_cat/data_headers/"+sim_name+"/redshifts.npy"))
chis_all = np.load(os.path.expanduser("~/repos/abacus_lc_cat/data_headers/"+sim_name+"/coord_dist.npy"))
zs_all[-1] = np.float('%.1f'%zs_all[-1])

# get functions relating chi and z
chi_of_z = interp1d(zs_all,chis_all)
z_of_chi = interp1d(chis_all,zs_all)

# load primary and secondary redshifts
zs_mt = np.load(os.path.expanduser("~/repos/abacus_lc_cat/data_mt/"+sim_name+"/zs_mt.npy"))

# obtain the vertices of the cube in box unit
vert = get_vertices_cube(units=Lbox/2.)

# centers of the cubes in Mpc/h
box0 = np.array([0.,0.,0.])-origin
box1 = np.array([0.,0.,Lbox])-origin
box2 = np.array([0.,Lbox,0.])-origin

# vertices of the three boxes
vert0 = box0+vert
vert1 = box1+vert
vert2 = box2+vert

# get functions relating chi and z
chi_of_z = interp1d(zs_all,chis_all)
z_of_chi = interp1d(chis_all,zs_all)

# minimax z
min_z = np.min(zs_all)
max_z = np.max(zs_all)
fs_sky = np.zeros(len(zs_mt))
chi_max = 3990. # Mpc/h
z_max = z_of_chi(chi_max)
print("z at chi_max = ", z_max)

# loop over all redshifts
for i in range(len(zs_mt)):
    # current redshift
    z = zs_mt[i]
    if z < min_z or z > max_z: continue
    
    # current chi
    chi = chi_of_z(z)
    print("chi = ", chi)

    if chi > 2*Lbox+100.: continue
    
    # create sphere of radius chi
    x_cart_z = x_cart*chi
    y_cart_z = y_cart*chi
    z_cart_z = z_cart*chi

    # which vertices are inside the dimensions of the box
    mask0 = is_in_cube(x_cart_z, y_cart_z, z_cart_z, vert0)
    mask1 = is_in_cube(x_cart_z, y_cart_z, z_cart_z, vert1)
    mask2 = is_in_cube(x_cart_z, y_cart_z, z_cart_z, vert2)
    mask = mask0 | mask1 | mask2
    #mask =  mask1 | mask2
    #mask = mask0
    
    # sky coverage in degrees
    f_sky = np.sum(mask)*full_sky/len(x_cart)
    print("f_sky [deg^2] = ", f_sky)
    fs_sky[i] = f_sky

    plt.axvline(x=z, color='k', ls='--')


np.save("/mnt/gosling1/boryanah/light_cone_catalog/sky_coverage/fs_sky.npy", fs_sky)
np.save("/mnt/gosling1/boryanah/light_cone_catalog/sky_coverage/zs_mt.npy", zs_mt)
print(zs_mt, fs_sky)
plt.axhline(y=full_sky/8, color='k', ls='--')
plt.axvline(x=z_max, color='g', ls='-')
plt.plot(zs_mt, fs_sky)
plt.xlabel("Redshift")
plt.ylabel("Sky coverage (deg^2)")
plt.xlim([0., 3.])
plt.show()
