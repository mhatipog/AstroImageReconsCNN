# Synthetic point-source response of the preserved 2017 3-CNN

This is an **exploratory astrophysical-fidelity check**, not a benchmark of
trained performance on real observational data.

The original paper reports PSNR, which measures image-averaged squared error.
A different question matters for astronomical use: when a faint source
appears, does the deconvolution retain its integrated brightness?

The experiment generates a known 11×11 Gaussian star profile (sigma=1.2px,
unit total flux) at 24 predetermined low-background sites in the
held-out M51a galaxy. Artificial sources are spaced by at least 80 pixels.

- Original 2017 trained M51a 3-CNN weights, unmodified.
- Same archived synthetic PSF family at `nr=5`.
- Same noise standard deviation `sigma=0.01`.
- Three injected source amplitudes: 1, 2 and 4 integrated normalized
  image-intensity units.
- Three independent Gaussian noise fields, paired between injected and
  control images.
- Circular aperture radii 8 and 16 pixels.

Both control and injected images have the **same** actual galaxy background
and **identical random noise**. Their predicted-image difference isolates
the CNN's *incremental response* to an artificial source. The measured
aperture flux ratio is `aperture(CNN(injected)-CNN(control)) /
injected_flux`. The same metric is computed directly on the degraded
observation as a physical control.

A recovery ratio of 1 would indicate perfect differential integrated flux
for the chosen aperture. Deviations indicate how this archived nonlinear
network redistributes or suppresses injected source flux.

## Scientific boundaries

- These injected Gaussian sources are synthetic, not PSF-calibrated physical
  stars. The test assesses a known mathematical perturbation only.
- The original DSS image is not an absolute photometric truth standard.
- All sites are from *one* galaxy and cannot be treated as independent
  astrophysical populations.
- No claims about false-positive detection rates or real survey photometry
  follow from this alone.
- The sites are the 24 locations with the lowest local maxima on a fixed
  80-pixel-spaced grid, ranked using the pristine reference **before**
  inspecting CNN predictions; no sites are excluded based on outcomes.

Run after installing the CPU dependencies documented in
`FULL_IMAGE_REPRODUCTION.md`, plus `astropy==7.1.0`:

```sh
python -m pytest -q tests/test_point_source_response.py
python point_source_response.py
```

Results: site coordinates, source-level trials and summary CSVs in
`reproduced/`. This branch is **research-only**, and does not modify the
upstream submission.
