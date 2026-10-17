# Scientific follow-up: PSF and noise sensitivity

**Research question.** How sensitive are the archived 3-CNN deconvolution
models from Flamary (EUSIPCO 2017) when the observational PSF and Gaussian
noise differ from their original training conditions?

The original experiment used six astronomical images with leave-one-image-out
training; a circular Airy PSF (`nr=5` in the supplied source) and fixed
Gaussian noise `sigma=0.01`. The experiment below changes only simulated
observation conditions; **no network is retrained or fine-tuned**.

## Protocol

- Held-out galaxies: M51a, M81 and M101. Each uses its own **original archived
  model trained on the other five images**.
- PSF: the original Airy function with `nr=4, 5, 6`, varying its spatial
  width. These are controlled deviations, not measured telescope PSFs.
- Gaussian noise: standard deviation `0.005, 0.01, 0.02` after the same
  full-image normalization and convolution as the original source.
- Three reproducible noise draws per galaxy; random noise fields are paired
  across all conditions to isolate parameter effects.
- Measured outcome: reconstructed PSNR, degraded-observation PSNR, and their
  difference, with the original 32-pixel evaluation crop.
- Total: **81 forward passes**, with no training, on CPU. Outputs are
  per-condition and nine-row summary CSVs.

This is a **small exploratory sensitivity study**, not evidence of
performance on real telescope observations. In particular, three galaxies
are not an adequate population sample; 3 seeds per galaxy do not turn them
into 9 independent objects, and this experiment does not demonstrate source
photometry or astrometric fidelity.

## Reproduce

Use the environment documented in `FULL_IMAGE_REPRODUCTION.md`, plus Astropy:

```bash
python -m pip install astropy==7.1.0
python -m pytest -q tests/test_psf_noise_sensitivity.py
python psf_noise_sensitivity.py
```

The GitHub Actions workflow runs the complete 81-case study and retains
`reproduced/psf_noise_sensitivity.csv` and
`reproduced/psf_noise_summary.csv` as downloadable artifacts.

## Relation to the published 2017 results

This is **not** a recalculation of the published Table I or a change to
Flamary's claims. The pretrained weights are unchanged. The 2017 result
remains the matched-PSF, matched-noise reference point. Any trends from this
test should be interpreted as a sensitivity observation and later validated
on independent sources and physically realistic noise/PSF models.

Reference: Rémi Flamary, [Astronomical image reconstruction with convolutional
neural networks](https://doi.org/10.23919/EUSIPCO.2017.8081654), 2017.
