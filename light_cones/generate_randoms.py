import os
import glob

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from tools import *

np.random.seed(300)

# all redshifts, steps and comoving distances of light cones files; high z to low z
zs_all = np.load("data_headers/redshifts.npy")
chis_all = np.load("data_headers/coord_dist.npy")
zs_all[-1] = np.float('%.1f'%zs_all[-1])

# get functions relating chi and z
chi_of_z = interp1d(zs_all,chis_all)
z_of_chi = interp1d(chis_all,zs_all)


def gen_rand(N_gals, gals_chis, fac=60, Lbox=2000., remove_edges=False, box_offset=True, origin=np.array([-990, -990, -990])):
    # number of randoms to generate is 8 times (octant) times number of galaxies
    #fac = 4*15 # 4 because we are doing upper half
    N_rands = fac*N_gals

    # generate randoms on the unit sphere (should be simple since theta and phi are arbitrary)
    #costheta = np.random.rand(N_rands)*2-1.
    #phi = np.random.rand(N_rands)*2*np.pi
    costheta = np.random.rand(N_rands)*1.01-0.01
    phi = np.random.rand(N_rands)*2*np.pi#np.pi/2.
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
    print(np.min(rands_chis), np.max(rands_chis))
    chi_cart = rands_chis

    # box size
    #Lbox = 2000.# Mpc/h

    # location of the observer relative to box centered
    #origin = np.array([-990,-990,-990])
    
    # vector between centers of the cubes and origin in Mpc/h (i.e. placing observer at 0, 0, 0)
    box0 = np.array([0., 0., 0.])-origin
    box1 = np.array([0., 0., Lbox])-origin
    box2 = np.array([0., Lbox, 0.])-origin
    
    # vertices of a cube centered at 0, 0, 0
    vert = get_vertices_cube(units=Lbox/2.)
    print(vert)

    if remove_edges:
        # tuks think about it more but TESTING    
        x_vert = vert[:, 0]
        y_vert = vert[:, 1]
        z_vert = vert[:, 2]
        vert[x_vert < 0, 0] += 10.
        vert[x_vert > 0, 0] -= 10.
        vert[y_vert < 0, 1] += 10.
        vert[z_vert < 0, 2] += 10.
    
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

    rands_pos = np.vstack((x_cart[mask], y_cart[mask], z_cart[mask], chi_cart[mask])).T

    
    rands_pos[:, :3] += origin
    if box_offset:
        rands_pos[:, :3] += Lbox/2.
    
    return rands_pos


def main():

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
    print("redshift range = ", gals_zs.min(), gals_zs.max())

    # downsampled NEW
    gals_down = np.load("/mnt/gosling1/boryanah/light_cone_catalog/AbacusSummit_base_c000_ph006/products/gals_down_"+model_no+".npy")
    gals_pos = gals_down[:, :3]
    gals_zs = gals_down[:, -1]

    # better not cause actually particles come from specific galaxies are moved by Lbox/2
    gals_norm, gals_chis, min_gals, max_gals = get_norm(gals_pos, np.array([10., 10., 10.]))
    #gals_chis = chi_of_z(gals_zs)
    N_gals = len(gals_chis)

    rands_pos = gen_rand(N_gals, gals_chis)
    #np.save("/mnt/gosling1/boryanah/light_cone_catalog/AbacusSummit_base_c000_ph006/products/randoms.npy", rands_pos)

