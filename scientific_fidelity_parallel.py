"""A preregistered controlled measurement-fidelity benchmark for the 2017 3-CNN.

All sources and test configurations are selected without inspecting predictions.
Classical methods are calibrated on synthetic, separate scenes and frozen.
This is a simulation benchmark; it is not evidence for real-survey performance.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import fftconvolve
from scipy.special import j1
from scipy.ndimage import gaussian_filter
from skimage.restoration import wiener, richardson_lucy

GALAXIES = ('M31', 'Hoag', 'M51a', 'M81', 'M101', 'M104')
PSF_RADII = (5, 4)
NOISE_MODELS = ('gaussian', 'photon_read')
SNR_LEVELS = (3, 5, 10, 20)
TEST_SEEDS = (111, 222)
CALIBRATION_SEEDS = (606, 707)
EXPOSURE = 3000.0
READ_NOISE = 0.003
GAUSSIAN_NOISE = 0.01
STAR_SIGMA = 1.2
STARS = 12
NULL_SITES = 12
APERTURE_RADIUS = 12
WINDOW_RADIUS = 24
SNR_RADIUS = 24

WIENER_CANDIDATES = (0.003, 0.03, 0.3)
RL_CANDIDATES = (5, 12, 25)


def airy(nr: int, n: int = 64) -> np.ndarray:
    axis = np.linspace(-1, 1, n)
    x, y = np.meshgrid(axis, axis)
    r = np.hypot(x, y) * np.pi * nr
    field = np.divide(j1(r), r, out=np.full_like(r, 0.5), where=r != 0)
    result = field ** 2
    return result / result.sum()


def star_kernel(sigma=STAR_SIGMA):
    yy, xx = np.mgrid[-6:7, -6:7]
    star = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return star / star.sum()


def make_sources(shape, sites, amplitudes, star=None):
    if star is None:
        star = star_kernel()
    out = np.zeros(shape, dtype=np.float64)
    for (y, x), amplitude in zip(sites, amplitudes):
        out[y - 6:y + 7, x - 6:x + 7] += amplitude * star
    return out


def choose_sites(reference, test_n=STARS, null_n=NULL_SITES):
    # Fixed geometry and one fixed rule using the clean reference. Spacing
    # between grid positions is 128 pixels; no outcome-based selection.
    candidates = []
    for y in range(160, 900, 128):
        for x in range(160, 900, 128):
            patch = reference[y-20:y+21, x-20:x+21]
            candidates.append((float(patch.std()), float(patch.mean()), y, x))
    candidates.sort()
    if len(candidates) < test_n + 2 * null_n:
        raise ValueError('Insufficient spatially separated candidates')
    # Interleave within the *reference-only* local-texture ranking to balance
    # the injected and two disjoint null groups (all use fixed grid sites).
    test = [(y, x) for _, _, y, x in candidates[0::3][:test_n]]
    null_cal = [(y, x) for _, _, y, x in candidates[1::3][:null_n]]
    null_test = [(y, x) for _, _, y, x in candidates[2::3][:null_n]]
    return test, null_cal, null_test


def translated_psf_template(psf, shape=(128, 128)):
    site = (shape[0]//2, shape[1]//2)
    field = make_sources(shape, [site], [1.0])
    convolved = fftconvolve(field, psf, mode='same')
    r = WINDOW_RADIUS
    return convolved[site[0]-r:site[0]+r+1, site[1]-r:site[1]+r+1]


def intrinsic_template():
    r = WINDOW_RADIUS
    size = 2*r+1
    output = np.zeros((size, size))
    output[r-6:r+7, r-6:r+7] = star_kernel()
    return output


def sample_noise(blurred, added, model, rng):
    """Paired exposures; photon field uses Poisson-superposition coupling."""
    if model == 'gaussian':
        noise = rng.normal(0, GAUSSIAN_NOISE, blurred.shape)
        base = blurred + noise
        with_sources = base + added
    elif model == 'photon_read':
        base_counts = rng.poisson(EXPOSURE * np.maximum(blurred, 0))
        injected_counts = rng.poisson(EXPOSURE * np.maximum(added, 0))
        read = rng.normal(0, READ_NOISE, blurred.shape)
        base = base_counts / EXPOSURE + read
        with_sources = (base_counts + injected_counts) / EXPOSURE + read
    else:
        raise ValueError('Unknown noise model: ' + model)
    return base.astype(np.float32), with_sources.astype(np.float32)


def nominal_amplitudes(blurred, sites, psf_template, noise_model, target_snr):
    r = WINDOW_RADIUS
    fluxes = []
    for y, x in sites:
        local = blurred[y-r:y+r+1,x-r:x+r+1]
        if noise_model == 'gaussian':
            variance = GAUSSIAN_NOISE ** 2
        else:
            variance = np.maximum(local, 0) / EXPOSURE + READ_NOISE ** 2
        # Gaussian matched-filter signal-to-noise with known *null* variance.
        # For Poisson injections, additional photons raise realized variance;
        # these are nominal background-limited SNRs, not measured SNRs.
        information = np.sum(psf_template ** 2 / variance)
        fluxes.append(target_snr / np.sqrt(information))
    return np.asarray(fluxes)


def aperture_weights(radius=APERTURE_RADIUS):
    yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
    return yy ** 2 + xx ** 2 <= radius ** 2


def response_flux(delta_image, site, radius=APERTURE_RADIUS):
    y, x = site
    w = aperture_weights(radius)
    return float(delta_image[y-radius:y+radius+1,x-radius:x+radius+1][w].sum())


def response_centroid(delta_image, site, radius=8):
    y, x = site
    yy, xx = np.mgrid[-radius:radius+1,-radius:radius+1]
    values = delta_image[y-radius:y+radius+1,x-radius:x+radius+1].astype(float)
    # Positive incremental signal centroid, *not* blind astrometry.
    weights = np.maximum(values,0)
    mass = weights.sum()
    if mass < 1e-10:
        return float('nan')
    return float(np.hypot((weights*yy).sum()/mass,(weights*xx).sum()/mass))


def fit_flux_local(image, site, template, weight=None):
    """At a known coordinate, fit source flux plus a local tilted background."""
    r = WINDOW_RADIUS
    y, x = site
    patch = image[y-r:y+r+1,x-r:x+r+1].astype(float)
    dy, dx = np.mgrid[-r:r+1,-r:r+1]
    X = np.column_stack((template.ravel(), np.ones(patch.size),
                         dx.ravel()/r, dy.ravel()/r))
    z = patch.ravel()
    if weight is not None:
        v = np.maximum(weight.ravel(),1e-12)
        inv_sqrt = np.sqrt(1/v)
        X = X * inv_sqrt[:, None]
        z = z * inv_sqrt
    coef, *_ = np.linalg.lstsq(X,z,rcond=None)
    return float(coef[0])


def methods(noisy, psf, params, cnn_layers=None):
    out = {'input': noisy}
    out['wiener'] = np.asarray(wiener(noisy, psf, balance=params['wiener'], clip=False),dtype=np.float32)
    out['rl'] = np.asarray(richardson_lucy(np.maximum(noisy, 0), psf,
                                          num_iter=int(params['rl']), clip=False),dtype=np.float32)
    if cnn_layers is not None:
        from full_image_inference import reconstruct
        out['cnn'] = reconstruct(noisy, cnn_layers, tile_size=196)
    return out


def synthetic_calibration_scene(seed=858):
    rng = np.random.default_rng(seed)
    n = 256
    y,x = np.mgrid[:n,:n]
    texture = gaussian_filter(rng.normal(size=(n,n)), 18)
    texture = texture / max(texture.std(),1e-10)
    sky = 0.18 + 0.00003*x + 0.00002*y + .006*texture
    sites = ((60,64),(64,184),(174,70),(186,184))
    return np.maximum(0,sky), sites


def calibrate_parameters():
    """Freeze classical settings using artificial backgrounds, not six galaxies."""
    calibration = []
    selections = {}
    for nr in PSF_RADII:
        psf = airy(nr)
        template = translated_psf_template(psf)
        for noise in NOISE_MODELS:
            losses = {('wiener',p):[] for p in WIENER_CANDIDATES}
            losses.update({('rl',p):[] for p in RL_CANDIDATES})
            for seed in CALIBRATION_SEEDS:
                sky,sites = synthetic_calibration_scene(seed)
                blurred = fftconvolve(sky, psf, mode='same')
                fluxes = nominal_amplitudes(blurred,sites,template,noise,5)
                injected = fftconvolve(make_sources(sky.shape,sites,fluxes),psf,mode='same')
                base, with_sources = sample_noise(blurred,injected,noise,
                                                 np.random.default_rng(seed+nr*100))
                for key,param in losses:
                    vals={'wiener':param,'rl':param}
                    b = methods(base,psf,vals)[key]
                    a = methods(with_sources,psf,vals)[key]
                    diff=a-b
                    for location,true_flux in zip(sites,fluxes):
                        recovered=response_flux(diff,location)/true_flux
                        losses[key,param].append(abs(recovered-1))
            chosen={}
            for method,grid in [('wiener',WIENER_CANDIDATES),('rl',RL_CANDIDATES)]:
                candidates=[]
                for param in grid:
                    score=float(np.mean(losses[method,param]))
                    calibration.append({'psf_nr':nr,'noise':noise,'method':method,
                                        'parameter':param,'calibration_mean_absolute_flux_error':score})
                    candidates.append((score,param))
                chosen[method]=min(candidates)[1]
            selections[f'{nr}_{noise}']=chosen
            print('CALIBRATION',nr,noise,json.dumps(chosen),flush=True)
    return calibration,selections


def load_clean(name):
    from astropy.io.fits import getdata
    raw=np.asarray(getdata(Path('data')/f'{name}.fits'),dtype=np.float64)
    normalized=np.maximum(raw/raw.max(),0)
    return normalized, normalized[1200:2224,1200:2224]


def load_model(name):
    from full_image_inference import load_layers
    return load_layers(Path('models')/(
        '32x32_10x10-64_6x6-16_5x5-1_PSF64_sigma0.010_'+name+'.h5'))


def evaluate(params, only_galaxy=None):
    """Test outcomes; never used for selection of classical parameters."""
    import torch
    torch.set_num_threads(2)
    results=[]
    for gi,name in enumerate(GALAXIES):
        if only_galaxy is not None and name != only_galaxy:
            continue
        full,clean=load_clean(name)
        sites,null_cal,null_test=choose_sites(clean)
        layers=load_model(name)
        print('GALAXY',name,'test sites',sites,'calibration nulls',null_cal,'held-out nulls',null_test,flush=True)
        for nr in PSF_RADII:
            psf=airy(nr)
            template=translated_psf_template(psf)
            blurred=fftconvolve(full,psf,mode='same')[1200:2224,1200:2224]
            for noise in NOISE_MODELS:
                frozen=params[f'{nr}_{noise}']
                for seed in TEST_SEEDS:
                    # For each SNR the base image uses a reproducible, same
                    # random seed; evaluate each injected exposure independently.
                    for snr in SNR_LEVELS:
                        fluxes=nominal_amplitudes(blurred,sites,template,noise,snr)
                        clean_sources=make_sources(clean.shape,sites,fluxes)
                        added=fftconvolve(clean_sources,psf,mode='same')
                        rng=np.random.default_rng(1000000+gi*10000+nr*1000+
                                                  (0 if noise=='gaussian' else 500)+seed*10+snr)
                        base,with_sources=sample_noise(blurred,added,noise,rng)
                        baseline=methods(base,psf,frozen,layers)
                        restored=methods(with_sources,psf,frozen,layers)
                        for method in ('input','wiener','rl','cnn'):
                            diff=restored[method]-baseline[method]
                            phot_template=template if method=='input' else intrinsic_template()
                            # Preregistered fixed-coordinate detection check:
                            # threshold fitted on one group of null positions
                            # then tested on disjoint null/source positions.
                            cal_fl=[fit_flux_local(baseline[method],pos,phot_template)
                                    for pos in null_cal]
                            center=float(np.median(cal_fl))
                            null_scale=1.4826*float(np.median(np.abs(np.asarray(cal_fl)-center)))
                            threshold=center+5*max(null_scale,1e-10)
                            null_false=sum(fit_flux_local(restored[method],pos,phot_template)>threshold
                                           for pos in null_test)
                            eval_pixels=np.s_[32:-32,32:-32]
                            truth=(clean+clean_sources)[eval_pixels]
                            im=(restored[method])[eval_pixels]
                            image_psnr=float(-10*np.log10(np.mean((im-truth)**2)))
                            for idx,(site,flux) in enumerate(zip(sites,fluxes)):
                                blind_est=fit_flux_local(restored[method],site,phot_template)
                                base_est=fit_flux_local(baseline[method],site,phot_template)
                                result={
                                    'galaxy':name,'psf_nr':nr,'noise':noise,
                                    'seed':seed,'target_snr':snr,'method':method,
                                    'site':idx,'flux_true':float(flux),
                                    'paired_aperture_response':response_flux(diff,site)/flux,
                                    'paired_centroid_error_px':response_centroid(diff,site),
                                    'blind_local_fit_flux_ratio':blind_est/flux,
                                    'blind_control_flux_ratio':base_est/flux,
                                    'fixed_site_detected_5mad':int(blind_est>threshold),
                                    'null_test_false_positive_count':null_false,
                                    'null_test_count':len(null_test),
                                    'image_psnr_db':image_psnr,
                                }
                                results.append(result)
                        print(f'CASE {name} nr={nr} noise={noise} seed={seed} SNR={snr}',flush=True)
        del full,clean,layers
    return results


def summarize(rows):
    outcomes=[]
    for nr in PSF_RADII:
        for noise in NOISE_MODELS:
            for snr in SNR_LEVELS:
                for method in ('input','wiener','rl','cnn'):
                    subset=[r for r in rows if r['psf_nr']==nr and r['noise']==noise
                            and r['target_snr']==snr and r['method']==method]
                    # The six held-out GALAXIES, not 12 source sites or noise
                    # realizations, are the independent clusters here.
                    galaxy_means=[]
                    galaxy_mae=[]
                    galaxy_blind_mae=[]
                    for name in GALAXIES:
                        group=[r for r in subset if r['galaxy']==name]
                        paired=np.array([r['paired_aperture_response'] for r in group])
                        galaxy_means.append(float(paired.mean()))
                        galaxy_mae.append(float(np.mean(np.abs(paired-1))))
                        galaxy_blind_mae.append(float(np.mean(np.abs(np.array([r['blind_local_fit_flux_ratio'] for r in group])-1))))
                    response=np.array(galaxy_means)
                    blind=np.array([r['blind_local_fit_flux_ratio'] for r in subset])
                    positions=np.array([r['paired_centroid_error_px'] for r in subset])
                    outcomes.append({
                        'psf_nr':nr,'noise':noise,'target_snr':snr,'method':method,
                        'n_galaxies':len(GALAXIES),'n_source_trials':len(subset),
                        'mean_galaxy_paired_flux_recovery':float(response.mean()),
                        'mean_abs_galaxy_flux_bias':float(np.mean(abs(response-1))),
                        'mean_absolute_paired_flux_error':float(np.mean(galaxy_mae)),
                        'mean_absolute_blind_flux_error':float(np.mean(galaxy_blind_mae)),
                        'galaxy_sd':float(response.std(ddof=1)),
                        'blind_local_fit_flux_ratio_median':float(np.median(blind)),
                        'paired_centroid_error_median_px':float(np.nanmedian(positions)),
                        'fixed_site_detection_rate':float(np.mean([r['fixed_site_detected_5mad'] for r in subset])),
                        'null_test_exceedance_rate':float(np.mean([r['null_test_false_positive_count']/r['null_test_count'] for r in subset[::STARS]])),
                        'mean_image_psnr_db':float(np.mean([r['image_psnr_db'] for r in subset[::STARS]])),
                    })
    return outcomes


def write_csv(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('calibrate','evaluate'),required=True)
    parser.add_argument('--selection-file',default='calibration_selection.json')
    parser.add_argument('--galaxy',choices=GALAXIES,default=None)
    args=parser.parse_args()
    if args.phase=='calibrate':
        rows,selection=calibrate_parameters()
        write_csv('results/calibration.csv',rows)
        Path(args.selection_file).write_text(json.dumps(selection,indent=2,sort_keys=True)+'\n')
        print('CALIBRATION_COMPLETE',json.dumps(selection,sort_keys=True),flush=True)
    else:
        selection=json.loads(Path(args.selection_file).read_text())
        print('FROZEN_SELECTION',json.dumps(selection,sort_keys=True),flush=True)
        rows=evaluate(selection, only_galaxy=args.galaxy)
        if args.galaxy:
            write_csv(f'results/{args.galaxy}/measurements.csv',rows)
            print('GALAXY_COMPLETE',args.galaxy,len(rows),flush=True)
            return
        write_csv('results/measurements.csv',rows)
        summary=summarize(rows)
        write_csv('results/summary.csv',summary)
        for s in summary:
            print('SUMMARY',json.dumps(s,sort_keys=True),flush=True)
        print('EVALUATION_COMPLETE',len(rows),len(summary),flush=True)

if __name__=='__main__':
    main()