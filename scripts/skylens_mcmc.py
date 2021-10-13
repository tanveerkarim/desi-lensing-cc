import os
os.environ["OMP_NUM_THREADS"] = "4" #need to be set early

import sys
import emcee

from skylens import *
from resource import getrusage, RUSAGE_SELF
import psutil
from distributed.utils import format_bytes

import numpy as np

from dask.distributed import Lock

import faulthandler; faulthandler.enable()
    #in case getting some weird seg fault, run as python -Xfaulthandler something.py
    # problem is likely to be in some package

import multiprocessing
from distributed import LocalCluster
from dask.distributed import Client

import pickle
import time

if __name__=='__main__':

    #We fix the following:
    #fix_cosmo : True means user provides cosmology; False means default cosmology
    #do_xi : whether to do real space correlation function
    #eh_pk : whether to use Eisenstein Hu power spectrum
    #use_binned_l : whether to do calculation in binned ℓ space
    
    outp = {} #save output
    test_run = True
    
    fix_cosmo=False 
    do_xi=False
    eh_pk=True 
    use_binned_l=False
    
    print('Doing mcmc',fix_cosmo,do_xi,use_binned_l,eh_pk,test_run) #err  True False True True     False True False True
    
    do_pseudo_cl=not do_xi
    use_binned_theta=use_binned_l
    
    #We fix parameters for the MCMC and `dask`:
    #- `test_run` : whether to run the shorter version for debugging purpose
    #- `nzbins` : number of redshift bins 
    #- `nwalkers` : number of walkers
    #- `nsteps` : number of sampled points/walker
    #- `python_file` : file that contains relevant survey parameters
    #- `ncpu` : number of cpus to be used for the calculation
    
    if test_run:
        nzbins=2
        nwalkers=25
        nsteps=50
        python_file='mcmc_test_args.py'
        ncpu=nwalkers
        
    LC,scheduler_info=start_client(ncpu=None,n_workers=ncpu,threads_per_worker=1,
                                   memory_limit='120gb',dashboard_address=8801,
                                   processes=True)
    client=client_get(scheduler_info=scheduler_info)
    print('client: ',client)
     
    lock = None
    
    ##define arguments for Skylens and set output file
    skylens_args=parse_python(file_name=python_file)
    skylens_args['do_xi']=do_xi
    skylens_args['do_pseudo_cl']=do_pseudo_cl
    skylens_args['use_binned_theta']=use_binned_theta
    skylens_args['use_binned_l']=use_binned_l
    
    if eh_pk:
        print('mcmc will use eh_pk')
        skylens_args['pk_params']['pk_func']='eh_pk'
        skylens_args['scheduler_info']=scheduler_info
        
        zs_bin=skylens_args['galaxy_zbins']
        file_home='/home/tkarim/SkyLens/scripts/tests/'
    
        if do_pseudo_cl:
            fname_out='mcmc_dat_pcl_{nz}_bl{bl}_bth{bth}_eh{eh_pk}_cgg_tst.pkl'.format(nz=zs_bin['n_bins'],
                                                                           bl=np.int(use_binned_l),
                                                                           bth=np.int(use_binned_theta),
                                                                           eh_pk=int(eh_pk))       
    get_cov=False
    try:
        fname_cl=file_home+fname_out
        with open(fname_cl,'rb') as of:
            cl_all=pickle.load(of)
        Win=cl_all['Win']
        if do_pseudo_cl:
            cl_cov=cl_all['cov']
            cov_inv=np.linalg.inv(cl_cov.todense())
            data=cl_all['pcl_b']

        zs_bin=cl_all['zs_bin']
        skylens_args['galaxy_zbins']=zs_bin
        print('read cl / cov from file: ',fname_cl)
    except Exception as err:
        get_cov=True
        print('cl not found. Will compute',fname_cl,err)
        
    if get_cov:    
        kappa0=Skylens(**skylens_args)
        print('kappa0 size',get_size_pickle(kappa0))
        print('kappa0.Ang_PS size',get_size_pickle(kappa0.Ang_PS))
    
        print('MCMC getting cl0G')
        cl0G=kappa0.cl_tomo()
        print('MCMC getting stack')
        cl_cov=client.compute(cl0G['stack']).result()
        cov_inv=np.linalg.inv(cl_cov['cov'].todense())
        data=cl_cov['pcl_b']
            
        Win=kappa0.Win
        outp['Win']=kappa0.Win
        outp['zs_bins']=kappa0.tracer_utils.z_bins
        with open(fname_cl,'wb') as of:
            pickle.dump(outp,of)
        del kappa0
        
    print(cl_cov['cov'])
    
    print('Got data and cov')
    if not np.all(np.isfinite(data)):
        x=np.isfinite(data)
        print('data problem',data[~x],np.where(x))
        
    ##Recalculate `kappa0` using predefined window and point the parallelized processes in `dask` to the same data matrices.
    
    Win['cov']=None
    skylens_args['do_cov']=False
    skylens_args['Win']=Win
    
    kappa0=Skylens(**skylens_args)
    Win=client.gather(kappa0.Win)
    kappa0.gather_data()
    
    ##passing data to parallelized instances
    data=client.scatter(data,broadcast=True)
    cov_inv=client.scatter(cov_inv,broadcast=True)
    
    ##parameter naming
    cosmo_fid=kappa0.Ang_PS.PS.cosmo_params
    params_order=['b1_{i}'.format(i=i) for i in np.arange(kappa0.tracer_utils.z_bins['galaxy']['n_bins'])]#,'Ase9','Om']
    
    ##set prior bounds on parameters
    priors_max=np.ones(len(params_order))*2
    priors_min=np.ones(len(params_order))*.5
    if not fix_cosmo:
        params_order+=['Ase9','Om', 'mnu']
        pf=np.array([cosmo_fid[k] for k in ['Ase9','Om', 'mnu']])
        priors_max=np.append(priors_max,pf*2)
        priors_min=np.append(priors_min,pf*.5)
        
    priors_min[-1] = 0.06 #set mnu limits explicitly
    priors_max[-1] = 1.

    ##`deepcopy` explicitly makes copies of data and does not act as pointers. Deleting window values because they take too much memory.
    
    zs_bin1=copy.deepcopy(client.gather(kappa0.tracer_utils.z_bins))
    zs_bin=copy.deepcopy(zs_bin1)
    del_k=['window','window_cl']
    for k in del_k:
        if zs_bin1['galaxy'].get(k) is not None:
            del zs_bin1['galaxy'][k]
        for i in np.arange(zs_bin1['galaxy']['n_bins']):
            if zs_bin1['galaxy'][i].get(k) is not None:
                del zs_bin1['galaxy'][i][k]
    
    zs_bin1=scatter_dict(zs_bin1,scheduler_info=scheduler_info,broadcast=True) 
    cl_bin_utils=scatter_dict(kappa0.cl_bin_utils,broadcast=True)
    xi_bin_utils=scatter_dict(kappa0.xi_bin_utils,broadcast=True)
    
    ##`fix_cosmo` sets only one cosmology. If `False`, then it recalculates the power spectrum every time. 
    if fix_cosmo:
        kappa0.Ang_PS.angular_power_z()
    else:
        kappa0.Ang_PS.reset()
    print('kappa0 pk',kappa0.Ang_PS.PS.pk_func)
    
    kappa0=client.scatter(kappa0,broadcast=True)
        
    proc = psutil.Process()
    print('starting mcmc ', 'mem, peak mem: ',format_bytes(proc.memory_info().rss),
          int(getrusage(RUSAGE_SELF).ru_maxrss/1024./1024.))
    
    ##define functions 
    def get_priors(params):#assume flat priors for now
        x=np.logical_or(np.any(params>priors_max,axis=1),np.any(params<priors_min,axis=1))
        p=np.zeros(len(params))
        p[x]=-np.inf
        return p
    
    def assign_zparams(zbins={},p_name='',p_value=None):
        pp=p_name.split('_')
        p_n=pp[0]
        bin_indx=np.int(pp[1])
        zbins[bin_indx][p_n]=p_value
        return zbins
    
    def get_params(params,kappa0,z_bins,log_prior):
        cosmo_params=copy.deepcopy(cosmo_fid)
        Ang_PS=kappa0.Ang_PS
        if not np.isfinite(log_prior):
            return cosmo_params,z_bins,Ang_PS
        zbins=copy.deepcopy(z_bins)
        i=0
        for p in params_order:
            if cosmo_params.get(p) is not None:
                cosmo_params[p]=params[i]
            else:
                zbins['galaxy']=assign_zparams(zbins=zbins['galaxy'],p_name=p,p_value=params[i])
            i+=1
        return cosmo_params,zbins,Ang_PS
    
    def get_model(params,data,cov_inv,kappa0,z_bins,log_prior,indx,Win,WT,WT_binned,cl_bin_utils,xi_bin_utils):
        """returns chisq given input parameters"""
        if not np.isfinite(log_prior):
            return -np.inf 
        cosmo_params,z_bins,Ang_PS=get_params(params,kappa0,z_bins,log_prior)
        
        model=kappa0.tomo_short(cosmo_params=cosmo_params,z_bins=z_bins,Ang_PS=Ang_PS,
                                Win=Win,WT=WT,WT_binned=WT_binned,
                                cl_bin_utils=cl_bin_utils,xi_bin_utils=xi_bin_utils)#,pk_lock=pk_lock)
        loss=data-model
        chisq=-0.5*loss@cov_inv@loss
        chisq+=log_prior
        return chisq #model
    
    def chi_sq(params,data,cov_inv,kappa0,z_bins,pk_lock):
        """returns parallelized chisq of given walkers"""
        t1=time.time()
        params=np.atleast_2d(params)
        log_priors=get_priors(params)
        n_params=len(params)
        models={}
        chisq={i:delayed(get_model)(params[i],data,cov_inv,kappa0,z_bins,log_priors[i],i,Win,WT,
                                    WT_binned,cl_bin_utils,xi_bin_utils) for i in np.arange(n_params)} #dask parallelization step
        chisq=client.compute(chisq).result()
        chisq=[chisq[i]for i in np.arange(n_params)]
        return chisq
    
    def ini_walkers():
        ndim=len(params_order)
        p0=np.zeros(ndim)
        p0f=np.zeros(ndim)
        i=0
        for p in params_order:
            if cosmo_fid.get(p) is not None:
                p0[i]=cosmo_fid[p]
                p0f=p0[i]*.5
            else:
                pp=p.split('_')
                p_n=pp[0]
                bin_indx=np.int(pp[1])
                p0[i]=zs_bin['galaxy'][bin_indx][p_n]
                p0f=.2
            i+=1
        return p0,p0f,ndim
    
    ##Define burning and thinning parameters
    nsteps_burn=5
    thin=2
    
    import time
    
    WT = None
    WT_binned = None
    
    def sample_params(fname=''):
        p00,p0_f,ndim=ini_walkers()
        p0 = np.random.uniform(-1,1,ndim * nwalkers).reshape((nwalkers, ndim))*p0_f*p00 + p00
    
        outp={}
        sampler = emcee.EnsembleSampler(nwalkers, ndim,chi_sq,threads=ncpu,a=2,vectorize=True,args=(data,cov_inv,kappa0,zs_bin1,lock))
                                                                    #a=2 default, 5 ok
    
        t1=time.time()
    
        pos, prob, state = sampler.run_mcmc(p0, nsteps_burn,store=False)
        print('done burn in '+str(time.time()-t1)+'  '+str((time.time()-t1)/3600.)+'  '+
        str(np.mean(sampler.acceptance_fraction)))
    
        sampler.reset()
    
        step_size=nsteps
        steps_taken=0

        pos, prob, state =sampler.run_mcmc(pos, step_size,thin=thin)
        steps_taken+=step_size
        outp['chain']=sampler.flatchain
        outp['p0']=p00
        outp['params']=params_order
        outp['ln_prob']=sampler.lnprobability.flatten()
        outp['acceptance_fraction']=np.mean(sampler.acceptance_fraction)
        outp['pos']=pos
        outp['prob']=prob
        outp['nsteps']=nsteps
        outp['nwalkers']=nwalkers
        outp['burnin']=nsteps_burn
        outp['thin']=thin
        outp['time']=time.time()-t1
    
        print('Done steps '+str(steps_taken)+ ' acceptance fraction ' +str(outp['acceptance_fraction'])+'  '
        'time'+str(time.time()-t1)+str((time.time()-t1)/3600.), 'nsteps: ',nsteps, 'chain shape',outp['chain'].shape)
        return outp
    
    #run sampling
    outp=sample_params()
    print('calcs done')
    
    #save values in output file
    outp['l0']=skylens_args['l']
    outp['l_bins']=skylens_args['l_bins']
    outp['do_xi']=do_xi
    outp['do_pseudo_cl']=do_pseudo_cl
    outp['use_binned_l']=use_binned_l
    outp['use_binned_theta']=use_binned_theta
    outp['data']=client.gather(data)
    outp['zbins']=skylens_args['galaxy_zbins']
    outp['cov_inv']=client.gather(cov_inv)
    outp['params_order']=params_order
    
    file_home='/home/tkarim/SkyLens/scripts/tests/'
    zs_bin1=client.gather(zs_bin1)

    if do_pseudo_cl:
        fname_out='pcl_{nz}_bl{bl}_bth{bth}_nw{nw}_ns{ns}_camb{fc}_cgg_tst.pkl'.format(nz=zs_bin1['galaxy']['n_bins'],
                                                                               bl=np.int(use_binned_l),
                                                                               bth=np.int(use_binned_theta),
                                                                               ns=nsteps,nw=nwalkers,
                                                                               fc=int(fix_cosmo))
    
    fname_out=file_home+fname_out
    with open(fname_out, 'wb') as f:
        pickle.dump(outp,f)
    print('file written: ',fname_out)
    client.shutdown()
    sys.exit(0)
