"""Audit the two possible Prox-TV parameter selection objectives.

Original run_deconv_all.py uses mse_crop(I, vc_recons_tv(I, ...)) where I is
the blurred/noisy input; Wiener and RL instead select against clean I0.
This bounded pilot compares both objectives using the *unmodified* original
deconv.vc_recons_tv method on 256x256 experimental subimages.
"""
from pathlib import Path
import csv
import numpy as np
import scipy.special
from astropy.io.fits import getdata
from scipy.io import loadmat
import deconv

GALAXIES=("M51a","M81","M101")
LAMBDAS=np.logspace(0,4,12)
ITERATIONS=100  # paper states 100; released run_deconv_all.py uses 500
CROP=slice(384,640)
INTERIOR=slice(32,-32)

def mse(a,b):
    return float(np.mean((a[INTERIOR,INTERIOR]-b[INTERIOR,INTERIOR])**2))

def psnr(err):
    return float(-10*np.log10(err))

def investigate():
    results=[]
    psf=deconv.get_PSF_airy(64,5)
    for galaxy in GALAXIES:
        fits=np.asarray(getdata(Path("data")/f"{galaxy}.fits"),dtype=np.float64)
        truth=np.maximum(0,fits/fits.max())[1200:2224,1200:2224][CROP,CROP]
        observation=loadmat(Path("res")/f"none_1024x1024_PSF64_sigma0.010_{galaxy}.mat")["Irec"][CROP,CROP]
        print(f"Galaxy {galaxy}: shape={truth.shape}",flush=True)
        for mu in LAMBDAS:
            reconstruction=deconv.vc_recons_tv(
                observation,psf,mu=float(mu),itermax=ITERATIONS)
            row={"galaxy":galaxy,"mu":float(mu),
                 "degraded_mse":mse(observation,reconstruction),
                 "true_mse":mse(truth,reconstruction)}
            row["true_psnr_db"]=psnr(row["true_mse"])
            results.append(row)
            print(f"TV galaxy={galaxy} mu={mu:.7g} mse_vs_degraded={row['degraded_mse']:.8g} mse_vs_truth={row['true_mse']:.8g} true_psnr={row['true_psnr_db']:.4f}",flush=True)
    return results

def analyze(results):
    audit=[]
    for galaxy in GALAXIES:
        rows=[r for r in results if r["galaxy"]==galaxy]
        by_noisy=min(rows,key=lambda r:r["degraded_mse"])
        by_truth=min(rows,key=lambda r:r["true_mse"])
        penalty=by_truth["true_psnr_db"]-by_noisy["true_psnr_db"]
        result={"galaxy":galaxy,
                "chosen_mu_source":by_noisy["mu"],
                "chosen_mu_clean":by_truth["mu"],
                "psnr_source_choice_db":by_noisy["true_psnr_db"],
                "psnr_clean_choice_db":by_truth["true_psnr_db"],
                "clean_oracle_gain_db":penalty}
        audit.append(result)
        print("AUDIT "+str(result),flush=True)
    return audit

def main():
    rows=investigate()
    audit=analyze(rows)
    directory=Path("reproduced");directory.mkdir(exist_ok=True)
    for path,data in [(directory/"tv_selection_curve.csv",rows),
                      (directory/"tv_selection_audit.csv",audit)]:
        with path.open("w",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=list(data[0]))
            writer.writeheader();writer.writerows(data)
    print(f"Completed {len(rows)} original-TV evaluations",flush=True)

if __name__=="__main__":
    main()
