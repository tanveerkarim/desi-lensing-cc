import numpy as np
import healpy as hp

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
        
nside = 2048#16384
npix = hp.nside2npix(nside)
ipix = np.arange(npix)
x_cart, y_cart, z_cart = hp.pix2vec(nside, ipix)
chi_max = 3990.# Mpc/h
Lbox = 2000.# Mpc/h

x_cart *= chi_max
y_cart *= chi_max
z_cart *= chi_max

origin = np.array([-990,-990,-990])

# centers of the cubes in Mpc/h
box0 = np.array([0.,0.,0.])-origin
box1 = np.array([0.,0.,2000.])-origin
box2 = np.array([0.,2000.,0.])-origin


vert = get_vertices_cube(units=Lbox/2.)
print(vert)

vert1 = box1+vert
vert2 = box2+vert
print(vert1)

mask1 = is_in_cube(x_cart,y_cart,z_cart,vert1)
mask2 = is_in_cube(x_cart,y_cart,z_cart,vert2)
mask = mask1 | mask2

print("f_sky [deg^2] = ",np.sum(mask)/len(x_cart)*41253.)

np.save("/mnt/store1/boryanah/AbacusSummit_base_c000_ph006/light_cones/mask_ring_%d.npy"%nside,mask)
#np.save("/global/common/software/desi/users/boryanah/light_cones/mask_ring_%d.npy"%nside,mask)
