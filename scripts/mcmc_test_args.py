import sys
from pandas import read_csv
from skylens.utils import *
from skylens.survey_utils import *
from cmbCrossELG import windowed_cl as wcl

"""tracer pairs for correlations"""
#tracers currently supported: galaxy,shear,kappa
corr_kk=('kappa','kappa')
corr_gg=('galaxy','galaxy')
corr_kg=('galaxy','kappa')

corrs=[corr_kk, corr_kg, corr_gg]
#corrs = [corr_gg]

"""cosmology and power spectra"""
from astropy.cosmology import Planck15 as cosmo
cosmo_params=dict({'h':cosmo.h,'Omb':cosmo.Ob0,'Omd':cosmo.Om0-cosmo.Ob0,'s8':0.817,'Om':cosmo.Om0,
                'Ase9':2.2,'mnu':cosmo.m_nu[-1].value,'Omk':cosmo.Ok0,'tau':0.06,'ns':0.965,
                'OmR':cosmo.Ogamma0+cosmo.Onu0,'w':-1,'wa':0,'T_cmb':cosmo.Tcmb0, 'Neff':cosmo.Neff})
cosmo_params['Oml']=1.-cosmo_params['Om']-cosmo_params['Omk']
pk_params={'non_linear':1,'kmax':30,'kmin':3.e-4,'nk':500,'scenario':'dmo','pk_func':'camb_pk_too_many_z','halofit_version':'takahashi'}

zmin = 0.1
zmax = 2.45104189964307
bg1 = 1.75
fsky = 0.38390517234802246 #DR8 Legacy Surveys

z_PS=np.linspace(zmin, zmax, 100) #redshifts at which to compute power spectra.
nz_PS=100 #number of redshifts to sample power spectra. used if z_PS is none
log_z_PS=2 #grid to generate nz_PS redshifts. 0==linear, 1==log, 2==log+linear. used if z_PS is none

"""C_ell"""
NSIDE = 1024
binsize = 100
#nbins = 20

lmax_cl = 2 * NSIDE
lmin_cl=0
l0=np.arange(lmin_cl,lmax_cl)

#following defines the ell bins. Using log bins in example, feel free to change.
lmin_cl_Bins=100
lmax_cl_Bins=lmax_cl-10
l_bins = np.arange(lmin_cl, lmax_cl, binsize)
#l_bins = np.logspace(lmin_cl, lmax_cl, nbins)
lb=0.5*(l_bins[1:]+l_bins[:-1])
l=l0

bin_cl=True #bin the theory and covaraince. 
window_lmax=2*lmax_cl #smaller value for testing. This should be 2X ell_max in the measurements.
use_binned_l=False  #FIXME: to speed up computation if using pseudo-cl inside mcmc. Needs to be tested. Leave it false for now.


"""window calculations"""
#relevant for pseudo_cl, correlation functions and covariances.
use_window=True #if you want to include the window effect. Code will return pseudo-cl and pseudo-cl covariance
#nside=16 #nside is not used by skylens.
window_l=None
#window_lmax=NSIDE #this is used to generate window_l=np.arange(window_lmax) if window_l is none. 
                    # for serious analysis, window_lmax=2*lmax_cl

store_win=True #store coupling matrices and other relevant quantities.
                #if False, these are computed along with the C_ell.
                #False is not recommended right now.
        
clean_tracer_window=True #remove tracer windows from memory once coupling matrices are done

wigner_files={} #wigner file to get pseudo_cl coupling matrices.
                #these can be gwenerated using Gen_wig_m0.py and Gen_wig_m2.py
                #these are large files and are stored as compressed arrays, using zarr package.
wig_home='/home/tkarim/SkyLens/temp/'
wigner_files[0]= wig_home+'dask_wig3j_l2048_w4096_0_reorder.zarr/'

"""covariance"""
do_cov=True # if you want to get covariance. Covariance is slow and this should be false if you are calling skylens inside mcmc.
SSV_cov=False # we donot have good model for super sample and tri-spectrum. We can chat about implementing some approximate analytical forms.
tidal_SSV_cov=False
Tri_cov=tidal_SSV_cov

sparse_cov=True #store covariances as sparse matrices
do_sample_variance=True #if false, only shot noise is used in gaussian covariance

#f_sky=0.35 #if there is no window. This can also be a dictionary for different correlation pairs, to account for partial overlaps, etc.
            # e.g. f_sky[corr_gg][(0,0)]=0.35, f_sky[corr_ggl][(0,0)]=0.1
            # for covariance, f_sky[corr_gg+corr_ggl][(0,0,0,1)]=0  (0,0,0,1)==(0,0)+(0,1)
        
"""generate simulated samples"""
galaxy_zbins= wcl.DESI_elg_bins(l=l, f_sky = fsky, nside = NSIDE, bg1 = bg1)
kappa_zbins= wcl.cmb_bins_here(l=l, nside = NSIDE, zmax = zmax)