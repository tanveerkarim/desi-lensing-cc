"""This file contains all the utility functions"""

import itertools
import numpy as np

def bin_mat(r=[],mat=[],r_bins=[]):
    """Sukhdeep's Code to bins data and covariance arrays

    Input:
    -----
        r  : array which will be used to bin data, e.g. ell values
        mat : array or matrix which will be binned, e.g. Cl values
        bins : array that defines the left edge of the bins,
               bins is the same unit as r

    Output:
    ------
        bin_center : array of mid-point of the bins, e.g. ELL values
        mat_int : binned array or matrix
    """

    bin_center=0.5*(r_bins[1:]+r_bins[:-1])
    n_bins=len(bin_center)
    ndim=len(mat.shape)
    mat_int=np.zeros([n_bins]*ndim,dtype='float64')
    norm_int=np.zeros([n_bins]*ndim,dtype='float64')
    bin_idx=np.digitize(r,r_bins)-1
    r2=np.sort(np.unique(np.append(r,r_bins))) #this takes care of problems around bin edges
    dr=np.gradient(r2)
    r2_idx=[i for i in np.arange(len(r2)) if r2[i] in r]
    dr=dr[r2_idx]
    r_dr=r*dr

    ls=['i','j','k','l']
    s1=ls[0]
    s2=ls[0]
    r_dr_m=r_dr
    for i in np.arange(ndim-1):
        s1=s2+','+ls[i+1]
        s2+=ls[i+1]
        r_dr_m=np.einsum(s1+'->'+s2,r_dr_m,r_dr)#works ok for 2-d case

    mat_r_dr=mat*r_dr_m
    for indxs in itertools.product(np.arange(min(bin_idx),n_bins),repeat=ndim):
        x={}#np.zeros_like(mat_r_dr,dtype='bool')
        norm_ijk=1
        mat_t=[]
        for nd in np.arange(ndim):
            slc = [slice(None)] * (ndim)
            #x[nd]=bin_idx==indxs[nd]
            slc[nd]=bin_idx==indxs[nd]
            if nd==0:
                mat_t=mat_r_dr[slc]
            else:
                mat_t=mat_t[slc]
            norm_ijk*=np.sum(r_dr[slc[nd]])
        if norm_ijk==0:
            continue
        mat_int[indxs]=np.sum(mat_t)/norm_ijk
        norm_int[indxs]=norm_ijk
    return bin_center,mat_int

def plot_powerspectra(cls, fields, ELL_min, ELL_max, **kwargs):
    """Calculates power spectra using anafast and produces plots against theory

    Input
    -----
        cls: list of cls to be plotted
        fields: 'gg', 'kk' or 'kg'
        ell_min: minimum ell to be plotted
        ell_max: max ell to be plotted
        label: plot legend label"""

    #bin cls
    _, cLs = bin_mat(ell, cls, bins)

    fltr_ELL = (ELL > ELL_min) & (ELL < ELL_max) #filter results between range

    #Plot C_L vs L
    if(fields == 'kk'):
        plt.loglog(ELL[fltr_ELL], cLs_cmb_th[fltr_ELL], 'rs', label = "CMB Theory")
        plt.ylabel(r"$C^{\kappa \kappa}_{L}$", fontsize = fs)
    elif(fields == 'gg'):
        plt.loglog(ELL[fltr_ELL], cLs_elg_th[fltr_ELL], "b*", label = "ELG Theory")
        plt.ylabel(r"$C^{g g}_{L}$", fontsize = fs)
    elif((fields == 'kg') | (fields == 'gk')):
        plt.loglog(ELL[fltr_ELL], cLs_cross_th[fltr_ELL], "mo", label = "CMB X ELG Theory")
        plt.ylabel(r"$C^{k g}_{L}$", fontsize = fs)

    plt.loglog(ELL[fltr_ELL], cLs[fltr_ELL], alpha = alpha, **kwargs)
    plt.xlabel(r"$L$", fontsize = fs)
    plt.legend(loc = 'best')
    plt.show()

    #Plot C_L_data/C_L_theory vs L
    if(fields == 'kk'):
        plt.plot(ELL[fltr_ELL], (cLs/cLs_cmb_th)[fltr_ELL], "rs", ls = '-')
    elif(fields == 'gg'):
        plt.plot(ELL[fltr_ELL], (cLs/cLs_elg_th)[fltr_ELL], "b*", ls = '-')
    elif((fields == 'kg') | (fields == 'gk')):
        plt.plot(ELL[fltr_ELL], (cLs/cLs_cross_th)[fltr_ELL], "mo", ls = '-')

    plt.xlabel(r"$L$", fontsize = fs)
    plt.ylabel(r"$\frac{C^{obs}_{L}}{C^{th}_{L}}$", fontsize = fs)
    plt.axhline(1)
    plt.show()
