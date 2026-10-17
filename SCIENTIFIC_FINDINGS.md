# Scientific measurement fidelity of the archived 2017 astronomical CNN

**Exploratory research memorandum — 17 October 2026**

**Reference.** Rémi Flamary, *Astronomical Image Reconstruction with Convolutional Neural Networks*, EUSIPCO 2017. These analyses use the unchanged six archived leave-one-image-out CNN weights. The original author has not been asked to endorse this new investigation. None of the findings should be interpreted as errors in the 2017 paper.

## Executive finding

The restored CNN reliably delivers the highest image-level PSNR of the tested methods under the nominal synthetic observation model. However, its source-response and *forced* source-photometry results depend on source brightness and image formation conditions. A narrow intrinsic-Gaussian template applied directly to its deconvolved output gives substantially less accurate flux measurements at higher input SNR than PSF fitting on the original blurred observation. That **does not mean the model simply discards that flux**: average differential aperture flux is nearly conserved at high source strength. The more plausible next hypothesis is an amplitude- and position-dependent **effective source response** that differs from the assumed intrinsic profile.

The distinction is relevant beyond this legacy CNN: excellent PSNR does not by itself establish reliable source measurements.

## What was actually executed

* 6 held-out original FITS galaxy fields, each with its own original archived 3-CNN trained on the other five.
* 2 Airy PSFs (`nr=5` matched and `nr=4` broadened), 2 noise processes (original Gaussian sigma=0.01, and a synthetic Poisson-exposure plus read-noise process), 4 nominal matched-filter source SNRs (3, 5, 10, 20), 2 held-out seeds.
* 12 preregistered source sites per field and 24 disjoint null positions per field, all chosen from the reference image without inspecting predictions.
* 192 simulated exposures; 9,216 method-level source measurements. All 6 images contributed 1,536 rows; none missing or nonfinite.
* Classical comparison: Wiener–Hunt, Richardson–Lucy and direct observation-domain PSF fitting. Their regularization/iteration settings were separately tuned on generated calibration fields and frozen *before* testing. Neither original weights nor original data were changed.

See `BENCHMARK_PROTOCOL.md` on the `science-six-galaxy-fidelity-benchmark` branch for the exact definitions, limitations, calibration and frozen decisions.

## Nominal matched-PSF, Gaussian-noise conditions

Image PSNR is measured against the clean reference plus injected sources on the paper's 32-pixel interior crop. Flux errors are mean absolute fractional errors, averaged equally across the six independent galaxies. Here 0.10 corresponds to a 10% error relative to the injected flux.

| SNR | Method | PSNR (dB) | Absolute blind fitted-flux error | Absolute paired-aperture response error |
|---:|---|---:|---:|---:|
| 5 | Original observation (PSF-fit) | 34.76 | 0.965 | 0.0755 |
| 5 | Wiener | 35.76 | 0.619 | 0.0290 |
| 5 | Richardson–Lucy | 37.31 | 0.741 | 0.0012 |
| 5 | **2017 CNN** | **38.50** | 0.702 | 0.0963 |
| 10 | Original observation (PSF-fit) | 34.75 | 0.476 | 0.0755 |
| 10 | Wiener | 35.76 | 0.525 | 0.0290 |
| 10 | Richardson–Lucy | 37.31 | 0.679 | 0.0015 |
| 10 | **2017 CNN** | **38.49** | 0.590 | 0.0750 |
| 20 | Original observation (PSF-fit) | 34.73 | 0.240 | 0.0755 |
| 20 | Wiener | 35.73 | 0.501 | 0.0290 |
| 20 | Richardson–Lucy | 37.27 | 0.663 | 0.0027 |
| 20 | **2017 CNN** | **38.46** | 0.503 | 0.0497 |

### Galaxy-clustered contrasts

The sampling unit is **galaxy**, not an injected star or repeated noise draw. The 95% intervals below are nonparametric bootstrap percentile intervals over the six galaxy-level contrasts (25,000 deterministic resamples). They are descriptive given the fixed six archival images, not population-level instrument uncertainties or multiplicity-adjusted confirmatory inference.

* Nominal Gaussian, SNR 20: CNN minus observation PSF-fit **absolute blind flux error = +0.263** (galaxy-bootstrap 95% interval **+0.117 to +0.378**). It is worse by at least 0.05 in **5/6 galaxies**, meeting the preregistered descriptive screening rule.
* Nominal Gaussian, SNR 10: same contrast **+0.114** (95% interval **−0.131 to +0.311**), also 5/6 galaxies worse by at least 0.05; this interval includes zero.
* Nominal Gaussian, SNR 5: CNN minus Richardson–Lucy **absolute paired-aperture error = +0.095** (95% interval **+0.080 to +0.107**), in the same direction for **all six galaxies**.
* Nominal Gaussian, SNR 20: CNN minus Richardson–Lucy **PSNR = +1.192 dB** (95% interval **+0.759 to +1.579 dB**), all six galaxies favor CNN.
* Broader PSF plus photon/read noise, SNR 20: CNN minus observation-domain PSF-fit **absolute blind flux error = +0.421** (95% interval **+0.284 to +0.534**), all six galaxies favor direct PSF-fitting. The CNN still produces somewhat higher PSNR than Richardson–Lucy (mean +0.222 dB).

## Crucial physical interpretation

**Do not equate poor intrinsic-template fitting with destroyed photon flux.** At nominal SNR 20, the CNN's *mean differential 12-pixel-aperture flux fraction* is **0.9955**, i.e., 99.55% on average, despite an average absolute fractional error across individual sites of ~4.97%. At SNR 5 its mean paired fraction is ~0.9099. The response is therefore amplitude-dependent under the tested conditions, but total flux at high SNR is *approximately* conserved by this paired diagnostic.

The very large apparent blind fitting errors result in part from fitting an **intrinsic 1.2-pixel Gaussian** to a network reconstruction whose empirical effective PSF has not been calibrated. On the blurred observation, the fit uses the correct blurred source template. The estimators therefore serve as a real workflow stress test but are **not equivalent calibrated photometric instruments**. The next decisive experiment should calibrate each method's effective response independently and then repeat **blind**, held-out flux measurements. Without that, no claim of intrinsic CNN photometric superiority or inferiority is justified.

### Detection and localization caveats

The threshold-exceedance metric uses a median-plus-five-MAD rule calibrated on 12 null sites and checked on 12 other fixed null sites. Nominal Gaussian SNR 10 produced *held-out null exceedances around 4–7%*, far higher than a calibrated 5-sigma Gaussian false-alarm probability. This reveals that galaxy structure and site-specific residuals dominate the simplistic local-null criterion. **These measurements must not be advertised as genuine survey detection completeness or false-positive rates.** A blinded detection experiment with proper background/PSF treatment and injected source catalogs is needed. The reported incremental centroid error uses source-free controls, so it is not a usable blind astrometry accuracy estimate.

## Decision and next discriminating study

This study justifies **continuing narrowly**, not publishing a claim or contacting Flamary about a supposed limitation.

1. **Calibrate the effective output PSF** for each restored model using separate simulated sources at known subpixel/14-pixel tiling phases, source levels and representative backgrounds. Freeze that calibration and evaluate flux on new independent injected sources. Test aperture, empirical-PSF and forward-observation photometry without using test controls.
2. **Run a genuine blind source detector** at pre-established null false-alarm rates with independent sky scenes, then compare detection completeness, positional error and flux bias across methods. Do not reuse calibration positions for evaluation.
3. **Strengthen baselines** using a validation-selected TV method and a modern deconvolution or direct pixel-level likelihood method; image PSNR and inference speed can be secondary.
4. **Validate physically** using calibrated instrument response, spatially varying PSFs, realistic background estimation and observational metadata, then consider a short research note if the central trade-off survives.

## Files and provenance

Original repo: https://github.com/rflamary/AstroImageReconsCNN  
Research branch: https://github.com/mhatipog/AstroImageReconsCNN/tree/science-six-galaxy-fidelity-benchmark  
Full successful GitHub Actions run: https://github.com/mhatipog/AstroImageReconsCNN/actions/runs/39026472290  
Archived experiment outputs: `measurements.csv` (9,216 rows), `summary.csv` (64 rows), `calibration.csv` (24 rows).  
Supplementary galaxy-clustered differences: `galaxy_clustered_contrasts.csv`.

**Status: exploratory observational-simulation experiment, not a peer-reviewed research result.**