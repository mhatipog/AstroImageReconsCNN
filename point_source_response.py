"""Paired artificial-point-source injection into held-out astronomical M51a.

A source-absent control and injected observation share exactly the same noise.
The difference between their reconstructions measures the archived CNN's
response to known added flux, independent of the galaxy background.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from astropy.io.fits import getdata
from scipy.signal import fftconvolve
from scipy.special import j1

from full_image_inference import load_layers, reconstruct

def airy_psf(n, nr):
    axis = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(axis, axis)
    radius = np.hypot(xx, yy) * np.pi * nr
    airy = np.divide(j1(radius), radius,
                     out=np.full_like(radius, 0.5), where=radius != 0)
    psf = airy**2
    return psf/psf.sum()

MODEL=Path("models/32x32_10x10-64_6x6-16_5x5-1_PSF64_sigma0.010_M51a.h5")
CROP=(slice(1200,2224),slice(1200,2224))
SEEDS=(0,1,2)
FLUXES=(1.0,2.0,4.0)
RADII=(8,16)

def blank_sky_centers(clean):
    # Explicitly exclude bright background structures. Predefined search
    # grid prevents cherry-picking based on the model's responses.
    candidates=[]
    for y in range(112,912,80):
        for x in range(112,912,80):
            local=clean[y-20:y+21,x-20:x+21]
            candidates.append((float(local.max()),y,x))
    # Rank by the original reference image only; do not tune a threshold
    # after seeing any model response. This is robust to plate background.
    candidates.sort()
    selected=candidates[:24]
    if len(selected)<8:
        raise ValueError("Too few predetermined grid locations")
    print(f"Reference local-max range for chosen positions: "
          f"{selected[0][0]:.5g} to {selected[-1][0]:.5g}",flush=True)
    return [(y,x) for _,y,x in selected]

def injected_field(full_shape,centers):
    field=np.zeros(full_shape,dtype=np.float64)
    dy,dx=np.mgrid[-5:6,-5:6]
    star=np.exp(-0.5*(dy**2+dx**2)/(1.2**2))
    star/=star.sum()
    for y,x in centers:
        y+=1200; x+=1200
        field[y-5:y+6,x-5:x+6]+=star
    return field

def aperture(image,y,x,radius):
    yy,xx=np.mgrid[-radius:radius+1,-radius:radius+1]
    cut=image[y-radius:y+radius+1,x-radius:x+radius+1]
    if cut.shape != (2*radius+1,2*radius+1):
        raise ValueError("Aperture outside image")
    return float(cut[(yy*yy+xx*xx)<=radius*radius].sum())

def run():
    torch.set_num_threads(2)
    original=np.asarray(getdata("data/M51a.fits"),dtype=np.float64)
    full=np.maximum(0,original/original.max())
    centers=blank_sky_centers(full[CROP])
    print(f"Selected {len(centers)} blank-sky positions before any model evaluations",flush=True)
    delta_field=injected_field(full.shape,centers)
    psf=airy_psf(64,5)
    blurred=fftconvolve(full,psf,mode="same")[CROP]
    delta_blurred=fftconvolve(delta_field,psf,mode="same")[CROP]
    layers=load_layers(MODEL)
    results=[]
    for seed in SEEDS:
        rng=np.random.default_rng(41000+seed)
        noise=rng.normal(0,0.01,(1024,1024))
        control=(blurred+noise).astype(np.float32)
        reconstructed_control=reconstruct(control,layers)
        for flux in FLUXES:
            observed=(control+flux*delta_blurred).astype(np.float32)
            reconstructed=reconstruct(observed,layers)
            response=reconstructed-reconstructed_control
            for y,x in centers:
                for radius in RADII:
                    # Aperture flux of the *difference* in reconstructed
                    # images, divided by injected total flux.
                    results.append({
                        "seed":seed,"injected_flux":flux,"radius":radius,
                        "y":y,"x":x,
                        "input_aperture_flux_ratio":
                            aperture(delta_blurred,y,x,radius),
                        "cnn_aperture_flux_ratio":
                            aperture(response,y,x,radius)/flux,
                    })
        print(f"Completed noise realization {seed}",flush=True)
    return results,centers

def summary(rows):
    outputs=[]
    for radius in RADII:
        for flux in FLUXES:
            group=[r for r in rows if r["radius"]==radius and r["injected_flux"]==flux]
            obs=np.array([r["input_aperture_flux_ratio"] for r in group])
            cnn=np.array([r["cnn_aperture_flux_ratio"] for r in group])
            result={
                "radius":radius,"flux":flux,"source_seed_pairs":len(group),
                "input_flux_fraction_mean":float(obs.mean()),
                "cnn_flux_fraction_mean":float(cnn.mean()),
                "cnn_flux_fraction_median":float(np.median(cnn)),
                "cnn_flux_fraction_sd":float(cnn.std(ddof=1)),
                "cnn_negative_fraction":float(np.mean(cnn<0)),
            }
            outputs.append(result)
            print("FLUX_SUMMARY "+json.dumps(result,sort_keys=True),flush=True)
    return outputs

def main():
    results,positions=run()
    outputs=summary(results)
    destination=Path("reproduced");destination.mkdir(exist_ok=True)
    for name,items in [("injection_sites.csv",[{"y":y,"x":x} for y,x in positions]),
                       ("source_flux_trials.csv",results),
                       ("source_flux_summary.csv",outputs)]:
        with (destination/name).open("w",newline="") as file:
            w=csv.DictWriter(file,fieldnames=list(items[0]))
            w.writeheader();w.writerows(items)
    print(f"Completed {len(results)} aperture measurements at {len(positions)} sites",flush=True)

if __name__=="__main__":
    main()
