"""Exploratory PSF/noise sensitivity of Flamary's original trained 3-CNNs.

Uses held-out astronomical images, original weights and the same forward model
family as EUSIPCO 2017. Changes only the simulated observation conditions.
This is an out-of-training-condition *stress test*, not a retrained benchmark.
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

GALAXIES = ("M51a", "M81", "M101")
RADII = (4, 5, 6)  # 2017 training PSF uses nr=5
SIGMAS = (0.005, 0.01, 0.02)  # 2017 training uses sigma=0.01
SEEDS = (0, 1, 2)
CROP = (slice(1200, 2224), slice(1200, 2224))
EVAL = (slice(32, -32), slice(32, -32))


def airy_psf(n=64, nr=5):
    positions = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(positions, positions)
    rr = np.hypot(xx, yy) * np.pi * nr
    airy = np.divide(j1(rr), rr, out=np.full_like(rr, 0.5), where=rr != 0)
    psf = airy ** 2
    return psf / psf.sum()


def psnr(prediction, original):
    a = np.asarray(prediction[EVAL], dtype=np.float64)
    b = np.asarray(original[EVAL], dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    return float(-10 * np.log10(mse))


def model_path(galaxy):
    return Path("models") / (
        "32x32_10x10-64_6x6-16_5x5-1_PSF64_sigma0.010_" + galaxy + ".h5"
    )


def study(galaxies=GALAXIES, radii=RADII, sigmas=SIGMAS, seeds=SEEDS):
    rows = []
    for gi, galaxy in enumerate(galaxies):
        raw = np.asarray(getdata(Path("data") / f"{galaxy}.fits"), dtype=np.float64)
        if raw.shape[0] < 2224 or raw.shape[1] < 2224:
            raise ValueError(f"{galaxy}: FITS dimensions unexpectedly small")
        # Follow the published training pipeline: normalize full image BEFORE
        # PSF convolution; then extract the central 1024x1024 experiment.
        truth = np.maximum(0, raw / raw.max())
        ground_truth = truth[CROP]
        layers = load_layers(model_path(galaxy))
        del raw
        print(f"Loaded galaxy={galaxy} full_shape={truth.shape}", flush=True)

        # Paired random fields: identical noise for a given galaxy/seed across
        # noise scales and PSF conditions. This isolates condition effects.
        noise_fields = {}
        for seed in seeds:
            rng = np.random.default_rng(20171000 + gi * 1000 + seed)
            noise_fields[seed] = rng.standard_normal((1024, 1024))
        for radius in radii:
            psf = airy_psf(n=64, nr=radius)
            blurred = fftconvolve(truth, psf, mode="same")[CROP]
            for sigma in sigmas:
                for seed in seeds:
                    observation = (blurred + sigma * noise_fields[seed]).astype(np.float32)
                    restored = reconstruct(observation, layers, tile_size=196)
                    row = {
                        "galaxy": galaxy,
                        "nr": radius,
                        "sigma": sigma,
                        "seed": seed,
                        "observation_psnr": psnr(observation, ground_truth),
                        "restored_psnr": psnr(restored, ground_truth),
                    }
                    row["gain_db"] = row["restored_psnr"] - row["observation_psnr"]
                    rows.append(row)
                    print(
                        f"CASE galaxy={galaxy} nr={radius} noise={sigma:.3f} "
                        f"seed={seed} observed={row['observation_psnr']:.3f} "
                        f"restored={row['restored_psnr']:.3f} "
                        f"gain={row['gain_db']:+.3f}",
                        flush=True,
                    )
            del blurred
        del truth, ground_truth, layers
    return rows


def aggregate(rows):
    results = []
    for radius in RADII:
        for sigma in SIGMAS:
            subset = [r for r in rows if r["nr"] == radius and r["sigma"] == sigma]
            if not subset:
                continue
            for key in ("observation_psnr", "restored_psnr", "gain_db"):
                values = np.array([r[key] for r in subset])
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite PSNR in {radius}, {sigma}: {key}")
            results.append({
                "nr": radius,
                "sigma": sigma,
                "samples": len(subset),
                "observation_psnr_mean": float(np.mean([r["observation_psnr"] for r in subset])),
                "restored_psnr_mean": float(np.mean([r["restored_psnr"] for r in subset])),
                "gain_db_mean": float(np.mean([r["gain_db"] for r in subset])),
                "gain_db_sd": float(np.std([r["gain_db"] for r in subset], ddof=1)),
            })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reproduced/psf_noise_sensitivity.csv"))
    parser.add_argument("--summary", type=Path, default=Path("reproduced/psf_noise_summary.csv"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    rows = study()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = aggregate(rows)
    with args.summary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print("SUMMARY_BEGIN", flush=True)
    for s in summaries:
        print("SUMMARY " + json.dumps(s, sort_keys=True), flush=True)
    print("SUMMARY_END", flush=True)
    print(f"Completed {len(rows)} held-out-image evaluations", flush=True)


if __name__ == "__main__":
    main()
