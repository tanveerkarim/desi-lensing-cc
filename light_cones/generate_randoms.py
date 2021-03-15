import os
import glob

import numpy as np
import healpy as hp

from scipy.interpolate import interp1d

def load_gals(fns,dim):    

    for fn in fns:
        tmp_arr = np.fromfile(fn).reshape(-1,dim)
        try:
            gal_arr = np.vstack((gal_arr,tmp_arr))
        except:
            gal_arr = tmp_arr
            
    return gal_arr

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

    print("f_sky [deg^2] = ",np.sum(mask)/len(x_pos)*41253.)
    
    return mask


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

    #if z > 1.6 or z < 0.8: continue
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
print("redshift range = ", gals_zs.min(), gals_zs.max())
gals_chis = chi_of_z(gals_zs)
N_gals = len(gals_chis)

# number of randoms to generate is 8 times (octant) times number of galaxies
fac = 8*15
N_rands = fac*N_gals

# generate randoms on the unit sphere (should be simple since theta and phi are arbitrary)
costheta = np.random.rand(N_rands)*2-1.
phi = np.random.rand(N_rands)*2*np.pi
theta = np.arccos(costheta)
x_cart = np.sin(theta)*np.cos(phi)
y_cart = np.sin(theta)*np.sin(phi)
z_cart = np.cos(theta)

# perhaps can only generate in the first octant

# get the redshifts/chis of the galaxies
rands_chis = np.repeat(gals_chis, fac)

# multiply the unit vectors by that
x_cart *= rands_chis
y_cart *= rands_chis
z_cart *= rands_chis

# box size
Lbox = 2000.# Mpc/h

# location of the observer
origin = np.array([-990,-990,-990])

# centers of the cubes in Mpc/h
box0 = np.array([0.,0.,0.])-origin
box1 = np.array([0.,0.,2000.])-origin
box2 = np.array([0.,2000.,0.])-origin

# vertices of a cube centered at 0, 0, 0
vert = get_vertices_cube(units=Lbox/2.)
print(vert)

# vertices for all three boxes
vert0 = box0+vert
vert1 = box1+vert
vert2 = box2+vert

# mask for whether or not the coordinates are within the vertices
mask0 = is_in_cube(x_cart, y_cart, z_cart, vert0)
mask1 = is_in_cube(x_cart, y_cart, z_cart, vert1)
mask2 = is_in_cube(x_cart, y_cart, z_cart, vert2)
mask = mask0 | mask1 | mask2
print("masked randoms = ", np.sum(mask)*100./len(mask))

rands_pos = np.vstack((x_cart[mask], y_cart[mask], z_cart[mask])).T
np.save("/mnt/gosling1/boryanah/light_cone_catalog/AbacusSummit_base_c000_ph006/products/randoms.npy", rands_pos)
