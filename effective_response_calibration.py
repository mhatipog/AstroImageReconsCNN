"""Independent empirical source-response calibration of restored 2017 CNNs.

Calibrate each model's output source template on synthetic backgrounds, then
freeze and test on each held-out astronomical image. No test truth or source
image is used for calibration.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import fftconvolve

from scientific_fidelity_benchmark import (
    GALAXIES, airy, choose_sites, synthetic_calibration_scene, load_model,
    load_clean, sample_noise, star_kernel, make_sources, nominal_amplitudes,
    translated_psf_template, intrinsic_template, fit_flux_local,
    WINDOW_RADIUS)
from full_image_inference import reconstruct

TEST_SNRS = (5, 10, 20)
TEST_SEEDS = (111, 222)
CALIBRATION_SNR = 10
PSF = airy(5)
OBS_TEMPLATE = translated_psf_template(PSF)


def calibrate_cnn_template(layers, seed=84292):
    background, _ = synthetic_calibration_scene(seed)
    # Well inside the field, far from source/edge.
    center = (128, 128)
    blurred = fftconvolve(background, PSF, mode='same')
    flux = nominal_amplitudes(blurred, [center], OBS_TEMPLATE,
                               'gaussian', CALIBRATION_SNR)[0]
    added = fftconvolve(make_sources(background.shape,[center],[flux]),PSF,mode='same')
    base, injected = sample_noise(blurred,added,'gaussian',np.random.default_rng(seed+100))
    clean_pred = reconstruct(base, layers, tile_size=196)
    injected_pred = reconstruct(injected, layers, tile_size=196)
    r=WINDOW_RADIUS
    empiric = (injected_pred-clean_pred)[center[0]-r:center[0]+r+1,
                                        center[1]-r:center[1]+r+1]/flux
    if empiric.shape != OBS_TEMPLATE.shape or not np.isfinite(empiric).all():
        raise ValueError('Invalid calibrated source response')
    return empiric.astype(np.float64)


def run():
    torch.set_num_threads(2)
    rows=[]
    for index,name in enumerate(GALAXIES):
        layers=load_model(name)
        empirical=calibrate_cnn_template(layers)
        full,clean=load_clean(name)
        sites,_,_=choose_sites(clean)
        blurred=fftconvolve(full,PSF,'same')[1200:2224,1200:2224]
        print('CALIBRATED',name,'empirical template integral',float(empirical.sum()),flush=True)
        for snr in TEST_SNRS:
            fluxes=nominal_amplitudes(blurred,sites,OBS_TEMPLATE,'gaussian',snr)
            added=fftconvolve(make_sources(clean.shape,sites,fluxes),PSF,'same')
            for seed in TEST_SEEDS:
                noise_rng=np.random.default_rng(700000+index*10000+snr*100+seed)
                base,observed=sample_noise(blurred,added,'gaussian',noise_rng)
                restored=reconstruct(observed,layers,tile_size=196)
                for i,(site,flux) in enumerate(zip(sites,fluxes)):
                    for method,image,template in [
                        ('direct_psf_fit',observed,OBS_TEMPLATE),
                        ('cnn_intrinsic_template',restored,intrinsic_template()),
                        ('cnn_empirical_template',restored,empirical),
                    ]:
                        estimate=fit_flux_local(image,site,template)
                        rows.append({
                            'galaxy':name,'snr':snr,'seed':seed,'site':i,
                            'method':method,'true_flux':float(flux),
                            'estimated_flux':estimate,'flux_ratio':estimate/flux,
                            'abs_fractional_error':abs(estimate/flux-1)
                        })
                print('EVAL',name,'snr',snr,'seed',seed,flush=True)
    return rows


def aggregate(rows):
    summary=[]
    methods=('direct_psf_fit','cnn_intrinsic_template','cnn_empirical_template')
    for snr in TEST_SNRS:
        for method in methods:
            sample=[x for x in rows if x['snr']==snr and x['method']==method]
            cluster=[]
            for galaxy in GALAXIES:
                vals=[x['abs_fractional_error'] for x in sample if x['galaxy']==galaxy]
                cluster.append(float(np.mean(vals)))
            result={'snr':snr,'method':method,'galaxies':len(GALAXIES),
                    'source_trials':len(sample),
                    'mean_absolute_flux_error':float(np.mean(cluster)),
                    'galaxy_sd':float(np.std(cluster,ddof=1)),
                    'median_flux_ratio':float(np.median([x['flux_ratio'] for x in sample]))}
            summary.append(result)
            print('CALIBRATED_SUMMARY',json.dumps(result,sort_keys=True),flush=True)
    return summary


def write(path, rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    rows=run()
    summary=aggregate(rows)
    write('results/effective_response_trials.csv',rows)
    write('results/effective_response_summary.csv',summary)
    print('EFFECTIVE_RESPONSE_COMPLETE',len(rows),flush=True)

if __name__=='__main__':
    main()