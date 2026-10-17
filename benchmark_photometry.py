"""Held-out astronomical source-photometry benchmark (exploratory).

All six original leave-one-galaxy-out models; injection strength calibrated to
the expected *observed-image* matched-filter SNR, not arbitrary pixel values.
Known-position flux fits on source-injected and untouched galaxy observations
are reported separately. No test-label-based tuning or model retraining.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from astropy.io.fits import getdata
from scipy.signal import fftconvolve
from scipy.special import j1

from full_image_inference import load_layers, reconstruct

GALAXIES = ("M31", "Hoag", "M51a", "M81", "M101", "M104")
SNR_LEVELS = (3, 5, 10, 20)
SIGMA = .01
CROP = (slice(1200, 2224), slice(1200, 2224))
R = 24

def airy(nr, n=64):
    xx, yy = np.meshgrid(np.linspace(-1,1,n), np.linspace(-1,1,n))
    r = np.hypot(xx, yy)*np.pi*nr
    a = np.divide(j1(r), r, out=np.full_like(r, .5), where=r!=0)
    p = a*a
    return p/p.sum()

def star():
    y,x = np.mgrid[-5:6,-5:6]
    k = np.exp(-(x*x+y*y)/(2*1.2**2))
    return k/k.sum()

def select_sites(clean, n=20):
    # Deterministic reference-only criterion; 80px spacing avoids overlap.
    candidates = []
    for y in range(144, 895, 80):
        for x in range(144, 895, 80):
            candidates.append((float(clean[y-30:y+31,x-30:x+31].max()),y,x))
    candidates.sort()
    return [(y,x) for _,y,x in candidates[:n]]

def inject(full_shape, sites):
    arr = np.zeros(full_shape, np.float32)
    k = star()
    for y,x in sites:
        arr[1200+y-5:1200+y+6,1200+x-5:1200+x+6] += k
    return arr

def local_template(kernel, y, x):
    return np.asarray(kernel[y-R:y+R+1,x-R:x+R+1],np.float64)

def fit_flux(image, site, template):
    y,x=site
    cut=np.asarray(image[y-R:y+R+1,x-R:x+R+1],np.float64)
    if cut.shape!=template.shape or cut.shape!=(2*R+1,2*R+1):
        raise ValueError("Improper template or edge cut")
    # Fit nuisance local constant + linear sky gradient, along with source
    # amplitude. This uses observation only, *not* clean galaxy background.
    yy,xx=np.mgrid[-R:R+1,-R:R+1]
    Q=np.stack((np.ones(cut.size), xx.ravel(), yy.ravel()),axis=1)
    t=template.ravel()
    z=cut.ravel()
    t=t-Q@np.linalg.lstsq(Q,t,rcond=None)[0]
    z=z-Q@np.linalg.lstsq(Q,z,rcond=None)[0]
    return float(t@z/(t@t))

def centroid_response(delta, y, x, radius=12):
    patch=np.asarray(delta[y-radius:y+radius+1,x-radius:x+radius+1],float)
    yy,xx=np.mgrid[-radius:radius+1,-radius:radius+1]
    # Compute only on positive incremental response. A signed response may
    # have cancelling lobes and no well-defined source centroid.
    positive=np.maximum(patch,0)
    weight=positive.sum()
    if weight<1e-9:
        return float("nan")
    return float(np.hypot((xx*positive).sum()/weight,
                          (yy*positive).sum()/weight))

def model_path(galaxy):
    return Path("models")/(
        f"32x32_10x10-64_6x6-16_5x5-1_PSF64_sigma0.010_{galaxy}.h5")

def run(galaxies, snrs, radii, seeds, sites_count, classical=False):
    torch.set_num_threads(2)
    results=[]
    for galaxy in galaxies:
        raw=np.asarray(getdata(f"data/{galaxy}.fits"),np.float64)
        if min(raw.shape)<2224 or not np.isfinite(raw).all():
            raise ValueError(f"Invalid archive FITS: {galaxy}")
        clean=np.maximum(0,raw/raw.max())
        sites=select_sites(clean[CROP], sites_count)
        point_field=inject(clean.shape,sites)
        weights=load_layers(model_path(galaxy))
        print(f"GALAXY {galaxy} sites={len(sites)}",flush=True)
        for nr in radii:
            psf=airy(nr)
            observation_base=fftconvolve(clean,psf,mode="same")[CROP]
            observed_source=fftconvolve(point_field,psf,mode="same")[CROP]
            # Matched-filter SNR calibration after convolution, on 49x49
            # patches, with the same 3-parameter sky nuisance projection.
            representative=local_template(observed_source,*sites[0])
            yy,xx=np.mgrid[-R:R+1,-R:R+1]
            Q=np.stack((np.ones(representative.size),xx.ravel(),yy.ravel()),axis=1)
            t=representative.ravel()
            t=t-Q@np.linalg.lstsq(Q,t,rcond=None)[0]
            norm=float(np.linalg.norm(t))
            if norm<=0: raise ValueError("Invalid source/PSF norm")
            unit_star=np.zeros((2*R+1,2*R+1),np.float64)
            unit_star[R-5:R+6,R-5:R+6]=star()
            for seed in seeds:
                rng=np.random.default_rng(67001+seed+len(galaxy)*100)
                noise=SIGMA*rng.standard_normal((1024,1024))
                null=(observation_base+noise).astype(np.float32)
                null_cnn=reconstruct(null,weights)
                if classical:
                    from skimage.restoration import wiener, richardson_lucy
                    # Fixed illustrative controls (not hyperparameter-tuned).
                    null_win=wiener(null,psf,balance=.01,clip=False)
                    null_rl=richardson_lucy(np.maximum(null,1e-6),psf,num_iter=10,clip=False)
                for snr in snrs:
                    flux=snr*SIGMA/norm
                    injected=(null+flux*observed_source).astype(np.float32)
                    cnn=reconstruct(injected,weights)
                    outputs=[("raw_psf_fit",injected,null,observed_source)]
                    outputs.append(("cnn_intrinsic_fit",cnn,null_cnn,None))
                    if classical:
                        win=wiener(injected,psf,balance=.01,clip=False)
                        rl=richardson_lucy(np.maximum(injected,1e-6),psf,num_iter=10,clip=False)
                        outputs.extend([("wiener_intrinsic_fit",win,null_win,None),
                                        ("rl_intrinsic_fit",rl,null_rl,None)])
                    for method,with_source,without_source,source_field in outputs:
                        for y,x in sites:
                            template=local_template(source_field,y,x) if source_field is not None else unit_star
                            detected=fit_flux(with_source,(y,x),template)
                            absent=fit_flux(without_source,(y,x),template)
                            row={
                                "galaxy":galaxy,"psf_nr":nr,"seed":seed,
                                "target_snr":snr,"injected_flux":flux,
                                "y":y,"x":x,"method":method,
                                "estimated_flux":detected,
                                "null_flux":absent,
                                "flux_ratio":detected/flux,
                                "paired_response_ratio":(detected-absent)/flux,
                                "centroid_error_px":centroid_response(
                                    with_source-without_source,y,x),
                            }
                            results.append(row)
                    print(f"CASE {galaxy} psf={nr} seed={seed} snr={snr} "
                          f"flux={flux:.6f}",flush=True)
        del clean,raw,point_field,weights
    return results

def summarize(rows):
    output=[]
    dimensions=sorted(set((x["psf_nr"],x["target_snr"],x["method"]) for x in rows))
    for nr,snr,method in dimensions:
        a=[r for r in rows if (r["psf_nr"],r["target_snr"],r["method"])==(nr,snr,method)]
        vals=np.asarray([r["estimated_flux"]/r["injected_flux"] for r in a])
        paired=np.asarray([r["paired_response_ratio"] for r in a])
        # Null-source exceedance and injected exceedance at same physical
        # 5-sigma flux limit. These counts are NOT calibrated FDR or power.
        null=np.asarray([r["null_flux"] for r in a])
        inj=np.asarray([r["estimated_flux"] for r in a])
        limits=np.asarray([5*r["injected_flux"]/r["target_snr"] for r in a])
        output.append({"psf_nr":nr,"snr":snr,"method":method,"n":len(a),
          "galaxies":len(set(r["galaxy"] for r in a)),
          "mean_flux_ratio":float(vals.mean()),
          "median_flux_ratio":float(np.median(vals)),
          "median_abs_relative_error":float(np.median(np.abs(vals-1))),
          "paired_mean_response":float(paired.mean()),
          "null_above_fixed_limit":int(np.sum(null>limits)),
          "injected_above_fixed_limit":int(np.sum(inj>limits)),
          "median_paired_centroid_px":float(np.nanmedian(
             [r["centroid_error_px"] for r in a]))})
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--galaxies",nargs="+",choices=GALAXIES,default=GALAXIES)
    p.add_argument("--snrs",type=int,nargs="+",default=SNR_LEVELS)
    p.add_argument("--psf",type=int,nargs="+",default=[5,4])
    p.add_argument("--seeds",type=int,nargs="+",default=[0,1])
    p.add_argument("--sites",type=int,default=20)
    p.add_argument("--classical",action="store_true")
    p.add_argument("--out",type=Path,default=Path("reproduced/science_benchmark"))
    a=p.parse_args()
    rows=run(a.galaxies,a.snrs,a.psf,a.seeds,a.sites,a.classical)
    sums=summarize(rows)
    a.out.mkdir(parents=True,exist_ok=True)
    for name,group in [("trials.csv",rows),("summary.csv",sums)]:
        with (a.out/name).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(group[0]))
            w.writeheader();w.writerows(group)
    for x in sums: print("SUMMARY "+json.dumps(x,sort_keys=True),flush=True)
    print(f"DONE {len(rows)} measurements, {len(sums)} groups",flush=True)

if __name__=="__main__":
    main()
