# Point-source photometry of the original leave-one-galaxy-out CNNs

**Research-only pilot**, separate from the Figure 3 correction PR. The source
models and input FITS data belong to Flamary (2017); cite the original paper.

## Question

Does a high-PSNR archived image reconstruction actually improve astronomical
point-source measurement, relative to direct fitting of the blurred image?

We run all six original trained three-layer models, each against the galaxy
left out of its original training set: M31, Hoag, M51a, M81, M101, M104.

An 11x11 normalized Gaussian artificial star is inserted into the *clean*
full-image field and blurred with the experiment's Airy PSF. Its amplitude
is set by analytical **matched-filter detection SNR 3, 5, 10, 20** under
sigma=0.01 independent additive Gaussian pixel noise. This SNR is a
simulation-defined ideal value, not a claim about detection confidence on
the galaxy background. The same reference-only, spatially separated
injection sites are used across conditions. The original CNN weights
are unchanged.

Two PSF conditions: original `nr=5` and broader mismatched `nr=4`.
Two independent noise seeds. The script supports a fixed, explicitly
*untuned* Wiener and Richardson-Lucy diagnostic (`--classical`); these
are **not competitive optimized baselines**. Do not rank or publish
classical comparisons without selecting their hyperparameters on
independent validation data.

The central comparison is:
1. **Direct observation PSF fitting**: least-squares amplitude for the known
   blurred star template, fitting a constant and linear local sky background.
2. **CNN + intrinsic star fitting**: the same least-squares estimator for the
   intrinsic artificial star on the reconstructed image.

Both use the *same* source-injected image, independently fitting and removing
a nuisance local background plane without subtracting a clean ground truth.
An **additional oracle diagnostic**, separately named
`paired_response_ratio`, subtracts each method's null-image flux estimate
from its injected-image flux estimate. It isolates source response but is
**not deployable photometry**. Don't confuse it with ordinary flux recovery.

Both injected and null-image amplitudes are retained in source-level CSVs.
Threshold counts use the same physical flux cut corresponding to expected
5-sigma input SNR, **not** calibrated false discovery rates, sensitivity,
or ROC metrics. Centroid errors are computed from the positive *paired
response* and are diagnostic, not blind astrometry.

## Controls and limitations

- No new training, fine-tuning, PSF estimation or human hand-picking sites.
- No tuning of CNN, detection threshold or other algorithms on test data.
- 20 lowest-local-maximum sites from a fixed 80px-spaced reference grid
  in each galaxy; these are **not guaranteed empty sky**.
- Each model evaluates a galaxy distinct from its own original train set,
  but the same six galaxies define this historical experiment; not
  independent new telescope data.
- At this stage **Gaussian noise only**; photon-counting/read noise and
  spatially varying PSFs require a separate validated forward model.
- Estimates at sites on the same galaxy are not independent samples.
  Aggregate statistically **by galaxy**, not by pixel/seed/site.
- The direct matched-filter control knows the true simulation PSF and star
  shape, whereas the 2017 CNN was trained for one PSF. This is a deliberately
  strong measurement baseline, not an unfair claim that one model wins.
- The results may demonstrate the case for measuring science outcomes instead
  of image PSNR, not a novel deconvolution architecture.

## Run

Use Python 3.11, the pinned environment from
`FULL_IMAGE_REPRODUCTION.md`, `astropy==7.1.0` and `pytest`.

```bash
python -m pytest -q tests/test_benchmark_photometry.py
python benchmark_photometry.py --galaxies M51a M81 --snrs 3 10 --psf 5 --sites 8
python benchmark_photometry.py --classical
```

The main full six-galaxy GitHub Actions job writes a trials CSV and aggregated
summary CSV. Results are experimental evidence, not an upstream bug report.
