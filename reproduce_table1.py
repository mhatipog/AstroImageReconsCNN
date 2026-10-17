"""Validate Flamary (EUSIPCO 2017) Table I using archived MATLAB outputs.

Postprocessing only; no CNN retraining or deconvolution is performed.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.io import loadmat

IMAGES = ('M31', 'Hoag', 'M51a', 'M81', 'M101', 'M104')
METHODS = ('wiener', 'rl', 'vc_tv', 'cnn0', 'cnn')
# Table I: https://doi.org/10.23919/EUSIPCO.2017.8081654
PUBLISHED_PSNR = np.array([
    [35.42, 35.11, 35.17, 35.82, 36.03],
    [37.29, 37.99, 36.66, 38.58, 39.99],
    [38.07, 38.26, 37.68, 38.96, 39.99],
    [35.91, 35.90, 35.38, 36.79, 37.26],
    [36.66, 37.79, 35.87, 38.31, 39.63],
    [36.01, 35.89, 35.34, 37.25, 38.10],
])
PUBLISHED_MEAN_PSNR = np.array([36.47, 36.65, 35.93, 37.48, 38.23])
PUBLISHED_MEAN_TIME = np.array([0.24, 1.15, 203.44, 0.52, 1.30])

def psnr(mse):
    values = np.asarray(mse, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError('MSE must be positive and finite')
    return -10.0 * np.log10(values)

def recover(directory):
    columns, times = [], []
    for method in METHODS:
        path = Path(directory) / f'{method}_1024x1024_PSF64_sigma0.010_all.mat'
        result = loadmat(path)
        mse = np.asarray(result['mse'], dtype=np.float64).ravel()
        elapsed = np.asarray(result['t'], dtype=np.float64).ravel()
        if mse.size < len(IMAGES) or elapsed.size < len(IMAGES):
            raise ValueError(f'{path}: fewer than six observations')
        columns.append(mse[:len(IMAGES)])
        times.append(elapsed[:len(IMAGES)])
    return np.column_stack(columns), np.column_stack(times)

def differences(mse, times):
    per_image = psnr(mse)
    # Published code: -10 log10(mean(MSE)), not mean(-10 log10(MSE)).
    aggregate = psnr(np.mean(mse, axis=0))
    mean_seconds = np.mean(times, axis=0)
    return (per_image, aggregate, mean_seconds,
            np.max(np.abs(per_image - PUBLISHED_PSNR)),
            np.max(np.abs(aggregate - PUBLISHED_MEAN_PSNR)),
            np.max(np.abs(mean_seconds - PUBLISHED_MEAN_TIME)))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--res-dir', type=Path, default=Path('res'))
    parser.add_argument('--csv', type=Path, default=Path('reproduced/table1.csv'))
    args = parser.parse_args()
    mse, times = recover(args.res_dir)
    per_image, aggregate, mean_seconds, dp, da, dt = differences(mse, times)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', *METHODS])
        for name, row in zip(IMAGES, per_image):
            writer.writerow([name, *[f'{v:.6f}' for v in row]])
        writer.writerow(['Pooled PSNR', *[f'{v:.6f}' for v in aggregate]])
        writer.writerow(['Archived mean time (s)', *[f'{v:.6f}' for v in mean_seconds]])
    print(f'Wrote {args.csv}')
    print(f'Max deviations from rounded Table I: PSNR {dp:.6f} dB; pooled PSNR {da:.6f} dB; mean time {dt:.6f} s')
    if dp > 0.011 or da > 0.011 or dt > 0.011:
        raise SystemExit('FAIL: saved result does not reproduce Table I to published precision')
    print('PASS: archived output reproduces Table I to published precision')

if __name__ == '__main__':
    main()
