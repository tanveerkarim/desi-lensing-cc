__all__ = ["set_window_here", "zbin_pz_norm", "source_tomo_bins", "DESI_elg_bins"]

import pandas as pd
import numpy as np
from skylens import *

def set_window_here(ztomo_bins_dict={}, nside=1024, mask = None, unit_win=False, cmb=False):
    """
        This function sets the window functions for the datasets. 
        These windows are necessary for converting cl to pseudo-cl.
    """
    #FIXME: make sure nside, etc. are properly matched. if possible, use same nside for cmb and galaxy maps. Use ud_grade where necessary.
    for i in np.arange(ztomo_bins_dict['n_bins']):
        if unit_win:
            cl_map=hp.ma(np.ones(12*nside*nside))
            cl_i=1
        elif cmb:
            #cl_map=np.load('/mnt/store1/tkarim/mask_2048.fits') #FIXME: add the CMB lensing window here.
            #window_map=np.load("/mnt/store1/boryanah/AbacusSummit_base_c000_ph006/light_cones/mask_edges_ring_2048.npy")
            #window_map=np.load("/home/tkarim/imaging-sys-covariance/dat/windows/Favg/Favg_map_unpickled.npy")
            #window_map = window_map.astype(np.float64)
            #window_map_noise = window_map
            #mask = cl_map
            print("cmb")
        else:
            window_map=np.load("/home/tkarim/imaging-sys-covariance/dat/windows/Favg/Favg_map_unpickled.npy") #randoms are the window function.
            #window_map=np.load("/mnt/store1/boryanah/AbacusSummit_base_c000_ph006/light_cones/mask_edges_ring_2048.npy") #randoms are the window function.
            window_map = window_map.astype(np.float64)
            window_map_noise = np.sqrt(window_map)
        
        if mask is None:
            mask=window_map>0 #FIXME: input proper mask if possible
        window_map[window_map<0]=0 #numerical issues can make this happen
        window_map/=window_map[mask].mean() #normalize window to 1
        window_map[~mask]=hp.UNSEEN
        window_map_noise[~mask]=hp.UNSEEN
        
        ztomo_bins_dict[i]['window']=window_map
        ztomo_bins_dict[i]['window_N']=window_map_noise #window of noise 

    return ztomo_bins_dict

def zbin_pz_norm(ztomo_bins_dict={},tomo_bin_indx=None,zbin_centre=None,p_zspec=None,ns=0,bg1=1,
                 mag_fact=0,k_max=0.3):
    """
        This function does few pre-calculations and sets some parameters for datasets that 
        will be input into skylens.
    """

    dzspec=np.gradient(zbin_centre) if len(zbin_centre)>1 else 1 #spec bin width

    if np.sum(p_zspec*dzspec)!=0:
        p_zspec=p_zspec/np.sum(p_zspec*dzspec) #normalize histogram
    else:
        p_zspec*=0
    nz=dzspec*p_zspec*ns

    i=tomo_bin_indx
    x= p_zspec>-1 #1.e-10; incase we have absurd p(z) values

    ztomo_bins_dict[i]['z']=zbin_centre[x]
    ztomo_bins_dict[i]['dz']=np.gradient(zbin_centre[x]) if len(zbin_centre[x])>1 else 1
    ztomo_bins_dict[i]['nz']=nz[x]
    ztomo_bins_dict[i]['ns']=ns
    ztomo_bins_dict[i]['W']=1. #redshift dependent weight
    ztomo_bins_dict[i]['pz']=p_zspec[x]*ztomo_bins_dict[i]['W']
    ztomo_bins_dict[i]['pzdz']=ztomo_bins_dict[i]['pz']*ztomo_bins_dict[i]['dz']
    ztomo_bins_dict[i]['Norm']=np.sum(ztomo_bins_dict[i]['pzdz'])
    ztomo_bins_dict[i]['b1']=bg1 # FIXME: this is the linear galaxy bias. Input proper values. We can also talk about adding other bias models if needed.
    ztomo_bins_dict[i]['bz1'] = None #array; set b1 to None if passing redz dependent bias 
    ztomo_bins_dict[i]['AI']=0. # this will be zero for our project
    ztomo_bins_dict[i]['AI_z']=0. # this will be zero for our project
    ztomo_bins_dict[i]['mag_fact']=mag_fact  #FIXME: You need to figure out the magnification bias prefactor. For example, see appendix B of https://arxiv.org/pdf/1803.08915.pdf
    ztomo_bins_dict[i]['shear_m_bias'] = 1.  #
    
    #convert k to ell
    zm=np.sum(ztomo_bins_dict[i]['z']*ztomo_bins_dict[i]['pzdz'])/ztomo_bins_dict[i]['Norm']
    #ztomo_bins_dict[i]['lm']=k_max*cosmo_h.comoving_transverse_distance(zm).value #not being used at the moment; if needed, talk to Sukhdeep 
    return ztomo_bins_dict

def source_tomo_bins(zphoto_bin_centre=None,p_zphoto=None,ntomo_bins=None,ns=26,
                     zspec_bin_centre=None,n_zspec=100,ztomo_bins=None,
                     f_sky=0.3,nside=256,use_window=False,
                    bg1=1,l=None,mag_fact=0,
                    k_max=0.3,use_shot_noise=True,**kwargs):
    """
        Setting galaxy redshift bins in the format used in skylens code.
        Need
        zbin_centre (array): redshift bins for every source bin. if z_bins is none, then dictionary with
                    with values for each bin
        p_zs: redshift distribution. same format as zbin_centre
        z_bins: if zbin_centre and p_zs are for whole survey, then bins to divide the sample. If
                tomography is based on lens redshift, then this arrays contains those redshifts.
        n_gal: number density for shot noise calculation
        n_zspec : number of histogram bins in spectroscopic dndz (if zspec_bin_centre is not passed)
        ztomo_bins : edges of tomographic bins in photometric redshift (assign galaxies to tomo bins using photz)
                    e.g. [0.6, 1., 1.6]
        k_max : cut in k-space; CHECK FOR BUG
    """
    ztomo_bins_dict={} #dictionary of tomographic bins

    if ntomo_bins is None:
        ntomo_bins=1

    if ztomo_bins is None:
        ztomo_bins=np.linspace(min(zphoto_bin_centre)-0.0001,max(zphoto_bin_centre)+0.0001,ntomo_bins+1)
    if zspec_bin_centre is None: #histogram of dndz; defines bin centres
        zspec_bin_centre=np.linspace(0,max(ztomo_bins)+1,n_zspec)
    dzspec=np.gradient(zspec_bin_centre)
    dzphoto=np.gradient(zphoto_bin_centre) if len(zphoto_bin_centre)>1 else [1]
    zphoto_bin_centre=np.array(zphoto_bin_centre)

    #zl_kernel=np.linspace(0,max(zbin_centre),50) #galaxy position kernel; identical to b*dndz
    #lu=Tracer_utils() 
    #cosmo_h=cosmo_h_PL #cosmology parameters in astropy convention; default is Skylens default

    zmax=max(ztomo_bins)

    l=[1] if l is None else l
    ztomo_bins_dict['SN']={} #shot noise dict
    ztomo_bins_dict['SN']['galaxy']=np.zeros((len(l),ntomo_bins,ntomo_bins)) # ell X no. of tomo bins X no. of tomo bins 
    ztomo_bins_dict['SN']['kappa']=np.zeros((len(l),ntomo_bins,ntomo_bins))

    for i in np.arange(ntomo_bins):
        ztomo_bins_dict[i]={}
        indx=zphoto_bin_centre.searchsorted(ztomo_bins[i:i+2]) #find bins that belong to this photometric bin

        if indx[0]==indx[1]: #if only one bin
            indx[1]=-1
        zbin_centre=zphoto_bin_centre[indx[0]:indx[1]]
        p_zspec=p_zphoto[indx[0]:indx[1]] #assuming spectroscopic and photometric dndz are same; CHANGE IF NOT 
        nz=ns*p_zspec*dzphoto[indx[0]:indx[1]]
        ns_i=nz.sum()

        ztomo_bins_dict = zbin_pz_norm(ztomo_bins_dict=ztomo_bins_dict, tomo_bin_indx=i, 
                                       zbin_centre=zbin_centre,
                                       p_zspec=p_zspec,ns=ns_i,bg1=bg1, mag_fact=mag_fact,k_max=k_max)
        
        zmax=max([zmax,max(ztomo_bins_dict[i]['z'])])
        if use_shot_noise:
            ztomo_bins_dict['SN']['galaxy'][:,i,i]=galaxy_shot_noise_calc(zg1=ztomo_bins_dict[i],
                                                                  zg2=ztomo_bins_dict[i])
            #the following is set in the CMB lensing bin
            #zs_bins['SN']['kappa'][:,i,i]=shear_shape_noise_calc(zs1=zs_bins[i],zs2=zs_bins[i],
            #                                                     sigma_gamma=sigma_gamma) #FIXME: This is almost certainly not correct

    ztomo_bins_dict['n_bins']=ntomo_bins #easy to remember the counts
    #ztomo_bins_dict['z_lens_kernel']=zl_kernel
    ztomo_bins_dict['zmax']=zmax
    ztomo_bins_dict['zp']=zphoto_bin_centre
    ztomo_bins_dict['pz']=p_zphoto
    ztomo_bins_dict['z_bins']=ztomo_bins
    
    if use_window:
        ztomo_bins_dict=set_window_here(ztomo_bins_dict=ztomo_bins_dict,nside=nside, unit_win=False)
    return ztomo_bins_dict

def cmb_bins_here(zs=1090,l=None,use_window=use_window,unit_win=False,nside=1024,zmax=2.45):
    """
    unit_win = boolean mask 
    nside = 2048; same as sims
    This function prepares the cmb lensing data into format required for input into skylens for theory predictions.
    """
    ztomo_bins_dict={}
    ztomo_bins_dict[0]={}

    ztomo_bins_dict=zbin_pz_norm(ztomo_bins_dict=ztomo_bins_dict,
                                 tomo_bin_indx=0,zbin_centre=np.atleast_1d(zs),
                                 p_zspec=np.atleast_1d(1),
                   ns=0,bg1=1)
    ztomo_bins_dict['n_bins']=1 #easy to remember the counts
    ztomo_bins_dict['zmax']=np.atleast_1d([zmax])
    #ztomo_bins_dict['zp_sigma']=0
    #ztomo_bins_dict['zp_bias']=0
    ztomo_bins_dict['nz']=1

    SN_read=np.genfromtxt('/mnt/store1/tkarim/cmb_lensing/data/MV/nlkk.dat',
                          names=('l','nl','nl2'))  #shot noise
    SN_intp=interp1d(SN_read['l'],SN_read['nl'],bounds_error=False, fill_value=0)      #FIXME: make sure using the correct noise power spectra.
    SN=SN_intp(l)
    SN *= 0 #DON'T DO THIS WHEN USING REAL DATA
#     SN=np.ones_like(l)
    ztomo_bins_dict['SN']={}
    ztomo_bins_dict['SN']['kappa']=SN.reshape(len(SN),1,1)
    if use_window:
        ztomo_bins_dict=set_window_here(ztomo_bins_dict=ztomo_bins_dict,
                                   nside=nside, cmb=True)
    return ztomo_bins_dict

def DESI_elg_bins(ntomo_bins=1, f_sky=0.3,nside=256,use_window=True, bg1=1, 
                       l=None, mag_fact=0,ztomo_bins=None,**kwargs):

    home='/home/tkarim/'
    fname='nz_blanc+abacus.txt'
#     t=np.genfromtxt(home+fname,names=True,skip_header=3)
    #t=np.genfromtxt(home+fname,names=True)
    t = pd.read_csv(home + fname)
    dz=t['Redshift_mid'][2]-t['Redshift_mid'][1]
    zmax=max(t['Redshift_mid'])+dz/2
    zmin=min(t['Redshift_mid'])-dz/2

    z=t['Redshift_mid']
    
    pz=t['dndz/deg^2']
    
    ns=np.sum(pz)
    d2r = 180/np.pi
    ns/=d2r**2 #convert from deg**2 to rd**2

    if ztomo_bins is None: #this defines the bin edges if splitting the sample into bins. Preferably pass it as an argument whenusing multiple bins.
        ztomo_bins=np.linspace(zmin, min(2,zmax), ntomo_bins+1) #define based on experiment
    print(zmin,zmax,ztomo_bins,ns)
    return source_tomo_bins(zphoto_bin_centre=z, p_zphoto=pz, ns=ns, ntomo_bins = ntomo_bins,
                            mag_fact=mag_fact, ztomo_bins=ztomo_bins,f_sky=f_sky,nside=nside,
                            use_window=use_window,bg1=bg1, l=l,**kwargs)