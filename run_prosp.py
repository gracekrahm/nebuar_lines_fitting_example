import numpy as np
import pandas as pd
from sedpy.observate import load_filters
import h5py
import prospect.io.read_results as pread
from prospect.models import priors, transforms
from prospect.models import priors_beta
from scipy.stats import truncnorm
from prospect.io import write_results as writer
from prospect.fitting import fit_model
import sys, os
zred=float(sys.argv[1])
#------------------------
# Convienence Functions
#------------------------
def find_nearest(array,value):
    idx = (np.abs(np.array(array)-value)).argmin()
    return idx

#----------------------
# SSP and noise functions
#-----------------------

def build_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    from prospect.sources import FastStepBasis
    sps = FastStepBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=compute_vega_mags)
    return sps


def build_noise(**extras):
    return None, None

#-------------------
# Build Model
#-------------------

def build_model(**kwargs):
    from prospect.models import priors, sedmodel
    from astropy.cosmology import FlatLambdaCDM
    from astropy import units as u
    print('building model')


    
    model_params = []

    #basics
    model_params.append({'name': "zred", "N": 1, "isfree": False,"init": zred})
    model_params.append({'name': 'pmetals', 'N': 1,'isfree': False,'init': -99,'prior': None})
    model_params.append({'name': 'imf_type', 'N': 1,'isfree': False,'init': 2})
    
    #M-Z
    model_params.append({'name': 'logmass', 'N': 1,'isfree': True,'init': 10.0,'prior': priors.Uniform(mini=7., maxi=12.)})
    model_params.append({'name': 'logzsol', 'N': 1,'isfree': True,'init': -0.5,'prior': priors.Uniform(mini=-1.9, maxi=0.3)})
    #SFH
    model_params.append({'name': "sfh", "N": 1, "isfree": False, "init": 3})
    model_params.append({'name': "mass", 'N': 9, 'isfree': False, 'init': 1., 'depends_on': transforms.logsfr_ratios_to_masses_psb})

    cosmo = FlatLambdaCDM(H0=68, Om0=0.3, Tcmb0=2.725)
    tuniv = cosmo.age(zred).to('Gyr').value
    nbins=9
    tbinmax = (tuniv * 0.85) * 1e9
    lim1, lim2 = 7.0, 8.0 #10 Myr and 30 Myr 

    agelims = np.array([1., lim1] +
                       np.linspace(lim2,np.log10(tbinmax), 5).tolist() +
                       np.linspace(np.log10(tbinmax), np.log10(tuniv*1e9), 4)[1:].tolist())
    model_params.append({"name": "agebins", 'N': 9, 'isfree': False, 'init': np.array([agelims[:-1], agelims[1:]]).T,
                           'depends_on': transforms.psb_logsfr_ratios_to_agebins})


    model_params.append({'name': 'tflex', 'N': 1, 'isfree': False, 'init': tuniv*0.5, 'units':'Gyr'})
    model_params.append({'name': 'nflex', 'N': 1, 'isfree': False, 'init': 5})
    model_params.append({'name': 'nfixed', 'N': 1, 'isfree': False, 'init': 3})
    model_params.append({'name': 'tlast', 'N': 1, 'isfree': True,
                                    'init': 0.08, 'prior': priors.TopHat(mini=.01, maxi=tuniv*0.5*0.75)})

    # These variables control the ratio of SFRs in adjacent bins
    # there is one for a fixed "youngest" bin, nfixed for nfixed "oldest" bins,
    # and (nflex-1) for nflex flexible bins in between
    model_params.append({'name': "logsfr_ratio_young", 'N': 1, 'isfree': True, 'init': 0.0, 'units': r'dlogSFR (dex)',
                                                 'prior': priors.StudentT(mean=0.0, scale=0.3, df=2)})
    model_params.append({'name': "logsfr_ratio_old", 'N': 3, 'isfree': True, 'init': np.zeros(3), 'units': r'dlogSFR (dex)',
                                               'prior': priors.StudentT(mean=np.zeros(3), scale=np.ones(3)*0.3, df=np.ones(3))})
    model_params.append({'name': "logsfr_ratios", 'N': 4, 'isfree': True, 'init': np.zeros(4), 'units': r'dlogSFR (dex)',
                                            'prior': priors.StudentT(mean=np.zeros(4), scale=0.3*np.ones(4), df=np.ones(4))})
    #Dust attenuation   
    model_params.append({'name': 'dust_type', 'N': 1,'isfree': False,'init': 0,'prior': None})
    model_params.append({'name': 'dust1', 'N': 1,'isfree': True, 'init': 1.0,'prior': priors.Uniform(mini=0., maxi=2.0)})
    model_params.append({'name': 'dust2', 'N': 1,'isfree': True, 'init': 1.0,'prior': priors.Uniform(mini=0.0, maxi=3.0)})
    model_params.append({'name': 'dust_index', 'N': 1,'isfree': True,'init': -0.9, 'prior': priors.Uniform(mini=-1.2, maxi=0.3)})
    #Dust Emission      
    model_params.append({'name': 'add_dust_emission', 'N': 1,'isfree': False,'init': 1})
    model_params.append({'name': 'duste_gamma', 'N': 1,'isfree': False,'init': 0.01,'prior': priors.Uniform(mini=0.0, maxi=1.0)})
    model_params.append({'name': 'duste_umin', 'N': 1,'isfree': False,'init': 10.0,'prior': priors.Uniform(mini=0.1, maxi=30.0)})
    model_params.append({'name': 'duste_qpah', 'N': 1,'isfree': False,'init': 1.,'prior': priors.Uniform(mini=0.0, maxi=10.0)})
    #Misc               
    model_params.append({'name': 'add_agb_dust_model', 'N': 1,'isfree': False,'init': 0})
    #Nebular Lines
    model_params.append({'name': 'add_neb_emission', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': r'log Z/Z_\odot',
                        'prior': None})
    model_params.append({'name': 'gas_logu', 'N': 1,
                        'isfree': False,
                        'init': -2.0,
                        'units': '',
                        'prior': priors.TopHat(mini=-4.0, maxi=-1.0)})

    model_params.append({'name': 'add_neb_continuum', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': r'log Z/Z_\odot',
                        'prior': None})


    model = sedmodel.SedModel(model_params)


    return model




#---------------------
# Setup Observations
#---------------------

#All of the sedpy filters
filternames = ['acs_wfc_f435w', 'acs_wfc_f475w', 'acs_wfc_f555w', 'acs_wfc_f606w', 'acs_wfc_f625w', 'acs_wfc_f775w', 'acs_wfc_f814w', 'acs_wfc_f850lp', 'bessell_B', 'bessell_I', 'bessell_R', 'bessell_U', 'bessell_V', 'cfht_megacam_gs_9401', 'cfht_megacam_is_9701', 'cfht_megacam_rs_9601', 'cfht_megacam_us_9301', 'cfht_megacam_zs_9801', 'cfht_wircam_H_8201', 'cfht_wircam_J_8101', 'cfht_wircam_Ks_8302', 'decam_Y', 'decam_g', 'decam_i', 'decam_r', 'decam_u', 'decam_z', 'gaia_bp', 'gaia_g', 'gaia_rp', 'galex_FUV', 'galex_NUV', 'herschel_pacs_100', 'herschel_pacs_160', 'herschel_pacs_70', 'herschel_spire_250', 'herschel_spire_350', 'herschel_spire_500', 'hipparcos_B', 'hipparcos_H', 'hipparcos_V', 'hsc_g', 'hsc_i', 'hsc_r', 'hsc_y', 'hsc_z', 'jwst_f070w', 'jwst_f090w', 'jwst_f1000w', 'jwst_f1130w', 'jwst_f115w', 'jwst_f1280w', 'jwst_f140m', 'jwst_f1500w', 'jwst_f150w', 'jwst_f162m', 'jwst_f1800w', 'jwst_f182m', 'jwst_f200w', 'jwst_f2100w', 'jwst_f210m', 'jwst_f250m', 'jwst_f2550w', 'jwst_f277w', 'jwst_f300m', 'jwst_f335m', 'jwst_f356w', 'jwst_f360m', 'jwst_f410m', 'jwst_f430m', 'jwst_f444w', 'jwst_f460m', 'jwst_f480m', 'jwst_f560w', 'jwst_f770w', 'jwst_moda_f070w', 'jwst_moda_f090w', 'jwst_moda_f115w', 'jwst_moda_f140m', 'jwst_moda_f150w', 'jwst_moda_f162m', 'jwst_moda_f182m', 'jwst_moda_f200w', 'jwst_moda_f210m', 'jwst_moda_f250m', 'jwst_moda_f277w', 'jwst_moda_f300m', 'jwst_moda_f335m', 'jwst_moda_f356w', 'jwst_moda_f360m', 'jwst_moda_f410m', 'jwst_moda_f430m', 'jwst_moda_f444w', 'jwst_moda_f460m', 'jwst_moda_f480m', 'jwst_modb_f070w', 'jwst_modb_f090w', 'jwst_modb_f115w', 'jwst_modb_f140m', 'jwst_modb_f150w', 'jwst_modb_f162m', 'jwst_modb_f182m', 'jwst_modb_f200w', 'jwst_modb_f210m', 'jwst_modb_f250m', 'jwst_modb_f277w', 'jwst_modb_f300m', 'jwst_modb_f335m', 'jwst_modb_f356w', 'jwst_modb_f360m', 'jwst_modb_f410m', 'jwst_modb_f430m', 'jwst_modb_f444w', 'jwst_modb_f460m', 'jwst_modb_f480m', 'keck_lris_Rs', 'keck_lris_g', 'mayall_mosaic_U_k1001', 'mayall_newfirm_H1', 'mayall_newfirm_H2', 'mayall_newfirm_J1', 'mayall_newfirm_J2', 'mayall_newfirm_J3', 'mayall_newfirm_K', 'mpgeso_wfi_B_eso842', 'mpgeso_wfi_Ic_eso845', 'mpgeso_wfi_Rc_eso844', 'mpgeso_wfi_U38_eso841', 'mpgeso_wfi_V_eso843', 'sdss_g0', 'sdss_i0', 'sdss_r0', 'sdss_u0', 'sdss_z0', 'sofia_hawc_bandA', 'sofia_hawc_bandB', 'sofia_hawc_bandC', 'sofia_hawc_bandD', 'sofia_hawc_bandE', 'spitzer_irac_ch1', 'spitzer_irac_ch2', 'spitzer_irac_ch3', 'spitzer_irac_ch4', 'spitzer_irs_16', 'spitzer_mips_160', 'spitzer_mips_24', 'spitzer_mips_70', 'spitzer_mips_70_dpe', 'stromgren_b', 'stromgren_u', 'stromgren_v', 'stromgren_y', 'subaru_moircs_H', 'subaru_moircs_J', 'subaru_moircs_Ks', 'subaru_suprimecam_B', 'subaru_suprimecam_Rc', 'subaru_suprimecam_V', 'subaru_suprimecam_ia427', 'subaru_suprimecam_ia445', 'subaru_suprimecam_ia464', 'subaru_suprimecam_ia484', 'subaru_suprimecam_ia505', 'subaru_suprimecam_ia527', 'subaru_suprimecam_ia550', 'subaru_suprimecam_ia574', 'subaru_suprimecam_ia598', 'subaru_suprimecam_ia624', 'subaru_suprimecam_ia651', 'subaru_suprimecam_ia679', 'subaru_suprimecam_ia709', 'subaru_suprimecam_ia738', 'subaru_suprimecam_ia767', 'subaru_suprimecam_ia797', 'subaru_suprimecam_ia827', 'subaru_suprimecam_ia856', 'subaru_suprimecam_ip', 'subaru_suprimecam_rp', 'subaru_suprimecam_zp', 'twomass_H', 'twomass_J', 'twomass_Ks', 'ukirt_wfcam_H', 'ukirt_wfcam_J', 'ukirt_wfcam_K', 'uvot_m2', 'uvot_w1', 'uvot_w2', 'vista_vircam_H', 'vista_vircam_J', 'vista_vircam_Ks', 'vista_vircam_Y', 'vista_vircam_Z', 'vlt_isaac_H', 'vlt_isaac_J', 'vlt_isaac_Ks', 'vlt_vimos_R', 'vlt_vimos_U', 'wfc3_ir_f105w', 'wfc3_ir_f110w', 'wfc3_ir_f125w', 'wfc3_ir_f140w', 'wfc3_ir_f160w', 'wfc3_uvis_f275w', 'wfc3_uvis_f336w', 'wfc3_uvis_f390w', 'wfc3_uvis_f475w', 'wfc3_uvis_f555w', 'wfc3_uvis_f606w', 'wfc3_uvis_f814w', 'wise_w1', 'wise_w2', 'wise_w3', 'wise_w4']


#------------------
# Build Observations
#-------------------


def build_obs(pd_dir,**kwargs):
    print('loading obs')
    import sedpy
    from astropy import units as u
    from astropy import constants
    from astropy.cosmology import FlatLambdaCDM    
    cosmo = FlatLambdaCDM(H0=68, Om0=0.3, Tcmb0=2.725)

    df = pd.read_csv(pd_dir)
    print(df.head())
    wav = df['wav']
    flux = df['spec']
    #wav  = np.asarray(wav)*u.micron #wav is in micron                                                                                                                             
    #wav = wav.to(u.AA)

    wav = np.asarray(wav)*u.AA
    lum = np.asarray(flux)*u.erg/u.s
#    dl = cosmo.luminosity_distance(zred).to('cm') #for mock seds from powderday

    dl = 1*u.cm #for mck seds from fsps (already redshifted)
    flux = lum/(4.*3.14*dl**2.)
    nu = constants.c.cgs/(wav.to(u.cm))
    nu = nu.to(u.Hz)
    flux /= nu
    flux = flux.to(u.Jy)
    maggies = flux / 3631.

    filters_unsorted = load_filters(filternames)
    waves_unsorted = [x.wave_mean for x in filters_unsorted]
    filters = [x for _,x in sorted(zip(waves_unsorted,filters_unsorted))]
    flx = []
    flxe = []
    redshifted_wav = wav #*(1.+zred) for powderday seds
    for i in range(len(filters)):
        flux_range = []
        wav_range = []
        for j in filters[i].wavelength:
            flux_range.append(maggies[find_nearest(redshifted_wav.value,j)].value)
            wav_range.append(redshifted_wav[find_nearest(redshifted_wav.value,j)].value)
        a = np.trapz(wav_range * filters[i].transmission* flux_range, wav_range, axis=-1)
        b = np.trapz(wav_range * filters[i].transmission, wav_range)
        flx.append(a/b)
        flxe.append(0.03* flx[i])
    flx = np.asarray(flx)
    flxe = np.asarray(flxe)
    flux_mag = flx
    unc_mag = flxe

    obs = {}
    obs['filters'] = filters
    obs['maggies'] = flux_mag
    obs['maggies_unc'] = unc_mag
    obs['phot_mask'] = np.isfinite(flux_mag)
    obs['wavelength'] = None
    obs['spectrum'] = None
    obs['pd_sed'] = maggies
    obs['pd_wav'] = redshifted_wav

    return obs



#-------------------
# Put it all together
#-------------------


def build_all(pd_dir,**kwargs):

    return (build_obs(pd_dir,**kwargs), build_model(**kwargs),
            build_sps(**kwargs), build_noise(**kwargs))




run_params = {'verbose':False,
              'debug':False,
              'output_pickles': True,
              'nested_bound': 'multi', # bounding method                                                                                      
              'nested_sample': 'auto', # sampling method                                                                                      
              'nested_nlive_init': 400,
              'nested_nlive_batch': 200,
              'nested_bootstrap': 0,
              'nested_dlogz_init': 0.05,
              'nested_weight_kwargs': {"pfrac": 1.0},
              }



if __name__ == '__main__':



    import os
    pd_dir = 'SED_zred8.csv'
    print('sed file:',pd_dir)
    obs, model, sps, noise = build_all(pd_dir,**run_params)
    run_params["sps_libraries"] = sps.ssp.libraries
    run_params["param_file"] = __file__
    hfile = f'fitted_SED_zred8.h5'
    print('Running fits')
    output = fit_model(obs, model, sps, noise, **run_params)
    print('Done. Writing now')
    writer.write_hdf5(hfile, run_params, model, obs,
              output["sampling"][0], output["optimization"][0],
              tsample=output["sampling"][1],
              toptimize=output["optimization"][1])


