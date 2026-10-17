"""Full-scale M51a investigation of archived Prox-TV reconstruction.

This uses the original 1024x1024 noisy image, the original FITS-based clean
reference, unmodified deconv.vc_recons_tv, 500 iterations, and candidate
regularization parameters from the *original* 20-point logspace grid.
It compares candidate predictions against the published saved reconstruction.
"""
from pathlib import Path
import csv
import numpy as np
import scipy.special
from scipy.io import loadmat
from astropy.io.fits import getdata
import deconv

NAME="M51a"
ITERS=500
CROP=32
# Two candidates surrounding the clean-optimal value from the pilot, plus
# the maximum grid value favored by the released parameter selection rule.
CANDIDATE_INDICES=(15,16,19)

def main():
    truth_full=np.asarray(getdata("data/M51a.fits"),dtype=np.float64)
    clean=np.maximum(0,truth_full/truth_full.max())[1200:2224,1200:2224]
    observed=np.asarray(loadmat("res/none_1024x1024_PSF64_sigma0.010_M51a.mat")["Irec"],dtype=float)
    archived_file=loadmat("res/vc_tv_1024x1024_PSF64_sigma0.010_M51a.mat")
    archived=np.asarray(archived_file["Irec"],dtype=float)
    psf=deconv.get_PSF_airy(64,5)
    if clean.shape!=observed.shape or observed.shape!=archived.shape:
        raise ValueError("Original FITS and stored 1024x1024 reconstruction mismatch")
    print("ARCHIVED", "PSNR", -10*np.log10(deconv.mse_crop(archived,clean,CROP)),
          "MSE",deconv.mse_crop(archived,clean,CROP),flush=True)

    rows=[]
    grid=np.logspace(0,4,20)
    for index in CANDIDATE_INDICES:
        mu=float(grid[index])
        print(f"START mu={mu:.8g} iterations={ITERS}",flush=True)
        restored=deconv.vc_recons_tv(observed,psf,mu=mu,itermax=ITERS)
        error_truth=deconv.mse_crop(restored,clean,CROP)
        error_input=deconv.mse_crop(restored,observed,CROP)
        error_archive=deconv.mse_crop(restored,archived,CROP)
        max_archive_error=float(np.abs(restored-archived).max())
        row={"mu":mu,"iterations":ITERS,"MSE_clean":float(error_truth),
             "PSNR_clean":float(-10*np.log10(error_truth)),
             "MSE_noisy_observation":float(error_input),
             "MSE_vs_archived_prediction":float(error_archive),
             "max_abs_vs_archived":max_archive_error}
        rows.append(row)
        print("CANDIDATE",row,flush=True)

    by_obs=min(rows,key=lambda r:r["MSE_noisy_observation"])
    by_clean=min(rows,key=lambda r:r["MSE_clean"])
    by_archive=min(rows,key=lambda r:r["MSE_vs_archived_prediction"])
    print("CHECK selected_noisy=",by_obs["mu"],
          "selected_clean=",by_clean["mu"],
          "best_match_archived=",by_archive["mu"],
          "clean_PSNR_difference_dB=",by_clean["PSNR_clean"]-by_obs["PSNR_clean"],flush=True)
    Path("reproduced").mkdir(exist_ok=True)
    with open("reproduced/tv_fullscale_m51a.csv","w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

if __name__=="__main__":
    main()
