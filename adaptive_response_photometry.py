"""Test observation-adaptive CNN source-response photometry.

No source-free test image, injected-source truth or reference galaxy values
are used to construct each adaptive template. We numerically perturb the
*observed* image along the known blurred source profile and measure the CNN's
local response. This is forced photometry at known simulated coordinates,
NOT a blind detection study.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import torch
from scipy.signal import fftconvolve
from scientific_fidelity_benchmark import (
    GALAXIES, airy, choose_sites, load_model, load_clean, sample_noise,
    make_sources, nominal_amplitudes, fit_flux_local, WINDOW_RADIUS,
    intrinsic_template, translated_psf_template,
)
from effective_response_calibration import calibrate_cnn_template
from full_image_inference import reconstruct

PSF=airy(5)
OBS_TEMPLATE=translated_psf_template(PSF)
EPSILON=0.05                   # fixed normalized integrated flux
SNRS=(5,10,20)
SEEDS=(333,444)               # not the previous calibration/evaluation draws
METHODS=("direct_psf","cnn_intrinsic","cnn_independent","cnn_adaptive")

def local_patch(array,site):
    r=WINDOW_RADIUS; y,x=site
    return np.asarray(array[y-r:y+r+1,x-r:x+r+1],dtype=np.float64)

def local_response(observed,base_output,layers,direction,epsilon=EPSILON):
    """Numerically linearize the restoration at the *observed* image."""
    if epsilon<=0: raise ValueError("epsilon must be positive")
    if direction.shape!=observed.shape:
        raise ValueError("perturbation must match observation shape")
    candidate=(np.asarray(observed,dtype=np.float64)
               + epsilon*np.asarray(direction,dtype=np.float64)).astype(np.float32)
    result=reconstruct(candidate,layers,tile_size=196)
    return (result.astype(np.float64)-base_output.astype(np.float64))/epsilon

def run():
    torch.set_num_threads(2)
    rows=[]
    for gi,name in enumerate(GALAXIES):
        layers=load_model(name)
        independent_template=calibrate_cnn_template(layers)
        full,clean=load_clean(name)
        sites,_,_=choose_sites(clean)
        blurred=fftconvolve(full,PSF,mode="same")[1200:2224,1200:2224]
        # One unit-flux perturbation simultaneously at non-overlapping sites.
        direction=fftconvolve(
            make_sources(clean.shape,sites,np.ones(len(sites))),PSF,mode="same"
        )
        for snr in SNRS:
            amplitudes=nominal_amplitudes(blurred,sites,OBS_TEMPLATE,"gaussian",snr)
            added=direction*0.0
            # Preserve site-dependent nominal fluxes even though Gaussian
            # pixel noise makes them close to equal.
            added=fftconvolve(make_sources(clean.shape,sites,amplitudes),
                               PSF,mode="same")
            for seed in SEEDS:
                rng=np.random.default_rng(940000+gi*10000+snr*100+seed)
                _,observed=sample_noise(blurred,added,"gaussian",rng)
                base_pred=reconstruct(observed,layers,tile_size=196)
                differential=local_response(observed,base_pred,layers,direction)
                for si,(site,flux) in enumerate(zip(sites,amplitudes)):
                    templates={
                        "direct_psf":OBS_TEMPLATE,
                        "cnn_intrinsic":intrinsic_template(),
                        "cnn_independent":independent_template,
                        "cnn_adaptive":local_patch(differential,site),
                    }
                    # Adaptive response may be close to zero in saturated or
                    # locally inactive ReLU regions; record instead of hiding.
                    adapted=np.asarray(templates["cnn_adaptive"])
                    degenerate=bool(np.linalg.norm(adapted)<1e-9)
                    for method in METHODS:
                        estimate=fit_flux_local(
                            observed if method=="direct_psf" else base_pred,
                            site,templates[method]
                        )
                        rows.append({
                            "galaxy":name,"snr":snr,"seed":seed,"site":si,
                            "y":site[0],"x":site[1],"method":method,
                            "true_flux":float(flux),"estimated_flux":estimate,
                            "flux_ratio":float(estimate/flux),
                            "abs_fractional_error":float(abs(estimate/flux-1)),
                            "adaptive_degenerate":int(degenerate),
                            "epsilon":EPSILON,
                        })
                print(f"TEST {name} snr={snr} seed={seed} sites={len(sites)}",
                      flush=True)
        del full,clean,layers,blurred,direction
    return rows

def summarize(rows):
    result=[]
    for snr in SNRS:
        for method in METHODS:
            a=[r for r in rows if r["snr"]==snr and r["method"]==method]
            galaxy_errors=[
                float(np.mean([r["abs_fractional_error"] for r in a
                               if r["galaxy"]==g])) for g in GALAXIES
            ]
            summary={
                "snr":snr,"method":method,"galaxies":len(GALAXIES),
                "trials":len(a),
                "mean_absolute_flux_error":float(np.mean(galaxy_errors)),
                "median_flux_ratio":float(np.median([x["flux_ratio"] for x in a])),
                "galaxy_sd":float(np.std(galaxy_errors,ddof=1)),
                "degenerate_fraction":float(np.mean([x["adaptive_degenerate"] for x in a]))
            }
            result.append(summary)
            print("ADAPTIVE_SUMMARY "+json.dumps(summary,sort_keys=True),flush=True)
        chosen=[]
        for g in GALAXIES:
            subset=[r for r in rows if r["galaxy"]==g and r["snr"]==snr]
            adapt=np.mean([r["abs_fractional_error"] for r in subset
                           if r["method"]=="cnn_adaptive"])
            fixed=np.mean([r["abs_fractional_error"] for r in subset
                           if r["method"]=="cnn_independent"])
            chosen.append(float(adapt-fixed))
        print("GALAXY_CONTRAST "+json.dumps({
          "snr":snr,"adaptive_minus_independent":chosen,
          "wins":int(sum(x<0 for x in chosen)),"mean":float(np.mean(chosen))
        },sort_keys=True),flush=True)
    return result

def write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def main():
    rows=run()
    summaries=summarize(rows)
    write("results/adaptive_response_trials.csv",rows)
    write("results/adaptive_response_summary.csv",summaries)
    print(f"ADAPTIVE_COMPLETE {len(rows)}",flush=True)

if __name__=="__main__":
    main()
