import numpy as np
import healpy as hp

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

    
    return mask


def get_norm(gals_pos, origin):
    gals_norm = gals_pos - origin
    vec_size = np.sqrt(np.sum((gals_norm)**2, axis=1))
    gals_norm /= vec_size[:, None]
    min_dist = np.min(vec_size)
    max_dist = np.max(vec_size)

    return gals_norm, vec_size, min_dist, max_dist


def get_ra_dec_chi(norm, chis):
    theta, phi = hp.vec2ang(norm)
    ra = phi
    dec = np.pi/2. - theta
    ra *= 180./np.pi
    dec *= 180./np.pi
    
    return ra, dec, chis
