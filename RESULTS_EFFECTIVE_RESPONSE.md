# Result: most of the apparent blind photometry bias was template mismatch

**Independent six-galaxy follow-up, 17 October 2026.**

The previous six-galaxy study showed excellent image PSNR but poor *naive* flux
measurements from the CNN when a narrow intrinsic Gaussian profile was fitted
directly to restored images. Yet paired differential aperture measurements
suggested approximately conserved source flux at higher SNR. This follow-up
tested whether a model-specific **effective source-response template**, measured
only on an independent generated calibration scene, could reconcile the two.

## Tested procedure

The protocol in `RESPONSE_CALIBRATION_PROTOCOL.md` was recorded before this
test. Each original archived leave-one-galaxy-out CNN was calibrated by one
known source at nominal SNR 10 inserted into an *independent synthetic* 256x256
sky image, using source-present minus source-free network output to obtain
a model-specific response template per unit input flux. These templates
were then frozen.

On all six original held-out galaxies, twelve distinct source sites, two
new noise seeds and nominal SNRs 5, 10 and 20 were evaluated under the
2017 matched Airy PSF and Gaussian noise. The measurement in each case was
a **local forced flux fit on the source-present image alone**—it did **not**
subtract the corresponding source-free galaxy. Exactly the same source
coordinates and background model were used for all three estimators.

## Results

Mean absolute fractional flux error, averaged equally across six galaxies
(0.20 means a 20% mean absolute error):

| Nominal SNR | Direct observation, correct PSF | CNN, naive intrinsic-Gaussian fit | CNN, independently calibrated response |
|---:|---:|---:|---:|
| 5 | 0.975 | 0.703 | 0.691 |
| 10 | 0.492 | 0.587 | **0.360** |
| 20 | 0.234 | 0.506 | **0.212** |

Calibration reduced CNN absolute flux error at SNR 20 in **all six galaxies**:
average improvement **0.293**, 95% galaxy-cluster bootstrap percentile
interval **[0.201, 0.371]** (25,000 deterministic resamples).

At SNR 10, the improvement was **0.227**, 95% interval **[0.086, 0.338]**,
in **five of six galaxies**.

At SNR 5, the average improvement was only **0.012**; two galaxies worsened
with the empirical template (especially M51a). This is not evidence that
calibration resolves the faintest-source problem.

The empirical CNN method had lower *average* error than direct PSF fitting
at SNR 10 and 20. **However, at SNR 20 it won on only three of six
individual galaxies.** Do not claim that CNN photometry is generally
superior on the strength of an averaged metric.

## Scientific meaning

This follow-up is evidence that a large portion of the earlier naive CNN
photometry bias was a **mismatched measurement template**, rather than
necessarily lost integrated source flux. CNN outputs should not be treated
as though an intrinsic 1.2-pixel Gaussian remains their true point-source
response.

However, one synthetic calibration source per network is insufficient for
a production-ready astronomical PSF or full photometric characterization:
the effective response can depend on source brightness, background,
subpixel position, overlapping 14-pixel CNN patch phase, noise and optical
PSF. The faint-source failure and M51a counterexample matter.

Next scientific test: calibrate a family of empirical response templates
over independent source strengths, patch phases and background textures,
then evaluate an unseen astronomical field and a *genuinely blind*
source-finding pipeline. Only if that survives independent data and
appropriate classical/modern controls should we consider a scientific
communication or manuscript.

## Provenance

Research branch:
https://github.com/mhatipog/AstroImageReconsCNN/tree/research-effective-response-calibration

Successful GitHub Actions experiment:
https://github.com/mhatipog/AstroImageReconsCNN/actions/runs/39027503612

Outputs: `effective_response_trials.csv` (1,296 method-level fits),
`effective_response_summary.csv` (9 summary rows). This is an **exploratory
simulation result**, not a peer-reviewed statement about real telescope data.
