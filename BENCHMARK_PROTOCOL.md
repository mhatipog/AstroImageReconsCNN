# Held-out astronomical measurement-fidelity benchmark

**Protocol fixed before evaluating the six held-out galaxies (17 October 2026).**

This is a small, controlled simulation study of the archived 2017 CNN by
Rémi Flamary, *Astronomical Image Reconstruction with Convolutional Neural
Networks* (EUSIPCO 2017). The publication's original results are
unmodified. No method is trained or fine-tuned on the benchmark test
injections.

**Research question:** Can a reconstruction achieve high image-wide PSNR
while distorting faint-source flux or location, especially if the
observation model differs from training?

## Independent units and treatments

- Six original FITS images: M31, Hoag, M51a, M81, M101 and M104. Each is
  evaluated with the archived CNN trained without that corresponding image.
- Airy PSF from the released 2017 code at `nr=5` (matched) or `nr=4`
  (broader than training).
- Original normalized Gaussian pixel noise at sigma 0.01, or a simulated
  photon-counting detector with exposure 3000 counts per normalized unit
  plus independent Gaussian read noise sigma 0.003.
- Nominal background-limited matched-filter source SNRs: **3, 5, 10, 20**.
  Flux is calculated per site from local null-image noise variance and
  the actual source-convolved PSF. For photon noise the injected photons
  increase actual variance, so these are *nominal*, not realized SNRs.
- Two independent held-out noise seeds (111 and 222) per configuration;
  injections and control observations share noise through a paired
  construction. For Poisson noise, independent source counts are added
  by Poisson superposition, rather than deterministically injecting
  expected counts.
- Source profile: normalized Gaussian with sigma 1.2 pixels, placed at
  12 predetermined sites per image, spaced by 128 pixels. Sites are
  selected **only** from low-texture positions in the clean reference,
  alternating within a 36-position fixed grid ranking with two disjoint
  groups of 12 null positions.
- Outcomes therefore use **six galaxies as independent clusters**, not
  individual source locations or two repeat realizations.

This yields 6 × 2 PSFs × 2 noise models × 2 noise seeds × 4 source SNRs =
**192 simulated test observations**, each with 12 injected sources,
plus paired source-absent controls. Four reconstruction/measurement
paths give **9,216 source-level method observations**.

## Methods, held fixed during testing

- **Observed image:** local tilted-background PSF-fitting photometry.
  No deconvolution. Uses the correct observation PSF.
- **Original archived 3-CNN:** unchanged weights for each held-out
  galaxy, reconstructed with the numerically validated modern inference.
- **Wiener–Hunt:** scikit-image `wiener` using the *known simulated PSF*,
  with clipping disabled.
- **Richardson–Lucy:** scikit-image `richardson_lucy` using the *known
  simulated PSF*, with clipping disabled; negative input values are
  clamped to zero as required by its nonnegative Poisson formulation.
  This formulation is imperfect for added Gaussian read noise.

Wiener balances (0.003, 0.03, 0.3) and RL iteration counts (5, 12, 25)
were assessed on **independent generated 256-pixel synthetic sky scenes**,
not on any of the six galaxy inputs. Two calibration seeds, four sources
and nominal SNR 5 were used. The score was mean absolute flux-response
error. The winning values, committed as `calibration_selection.json`,
are fixed for all test data and reproducible with `--phase calibrate`.

These methods are **not** claimed to be the globally optimal baselines.
A tuned total-variation method and newer learned methods are not yet
included; results must not be presented as a comprehensive method ranking.

## Measurements

Primary scientific measurements:

1. **Blind local fitted source flux** at the *known trial coordinates*,
   normalized by injected true flux. A linear local gradient and
   intercept are fitted along with source amplitude. The observation
   uses the blurred source template, while reconstructions use the
   intrinsic Gaussian source template. The latter may not represent
   the methods' effective PSFs, which is precisely a potential source
   of model-dependent photometric bias.
2. **Paired differential aperture flux:** subtract reconstruction of
   source-absent control from reconstruction with injected sources,
   integrate inside 12-pixel aperture and divide by injected true flux.
   This is an **oracle diagnostic** isolating model response, not a
   directly deployable source-flux estimator.
3. **Image PSNR**, against the clean galaxy plus injected ground-truth
   stars, with the same 32-pixel crop as the original paper.
4. **Paired incremental centroid error** relative to known source position
   (also an oracle diagnostic, **not** blind astrometry).
5. **Fixed-coordinate threshold exceedance**, using a threshold fitted
   on 12 calibration-null sites of the source-free output (median + five
   times robust median absolute deviation), evaluated at 12 injected and
   12 disjoint held-out null sites. This is **not** a full blind source
   finder, survey completeness, or a calibrated global false-discovery
   rate; galaxy backgrounds may violate the assumed null exchangeability.

## Pre-specified interpretation

The principal descriptive comparison is mean absolute *blind fitted*
flux error per galaxy, averaged equally over six galaxies, relative
to observation-domain PSF-fitting. Supporting outcomes are paired aperture
flux bias, PSNR, position error and null-site exceedances.

A credible warning signal is a worse flux-error metric for CNN than direct
PSF fitting by **at least 0.05 (absolute fraction)** across at least four
of six held-out galaxies, at one or more nominal SNRs, without a matching
advantage on the paired-response diagnostic. This is a descriptive screening
rule, **not** a significance threshold or confirmatory hypothesis test.
Analyze mismatched-PSF and photon-noise conditions independently.
Do not pool sites and repetitions as independent galaxies.

If results are ambiguous, extend to more independent real and simulated
sky fields, separate tuning-validation-test splits, instrument-specific
PSFs, background estimation and a proper blinded detection pipeline.
Only then consider research claims or approach the original author.

## Reproduction

Python 3.11, NumPy 2.4.6, SciPy 1.17.1, h5py 3.16.0,
scikit-image 0.26.0, Astropy 7.1.0, PyTorch 2.14.1+cpu.

```sh
python -m pip install numpy==2.4.6 scipy==1.17.1 h5py==3.16.0 scikit-image==0.26.0 astropy==7.1.0 pytest==9.1.1
python -m pip install 'torch==2.14.1+cpu' --index-url https://download.pytorch.org/whl/cpu
pytest -q tests/test_scientific_fidelity.py
python scientific_fidelity_benchmark.py --phase calibrate --selection-file /tmp/calibration.json
python scientific_fidelity_benchmark.py --phase evaluate --selection-file calibration_selection.json
```

GitHub Actions checks that the independent calibration still produces
the **pre-committed selection**, before it reads the held-out galaxies.
It then saves calibration, individual measurements and summary CSVs.

This study is separate from the upstream one-line Figure 3 correction.
