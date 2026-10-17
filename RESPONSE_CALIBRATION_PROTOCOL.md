# Can an independently calibrated effective source response remove the apparent CNN photometry bias?

**Hypothesis and analysis plan written before executing this follow-up.**

The six-galaxy benchmark revealed a paradox: original CNN predictions have high
image PSNR and approximately conserved mean *paired aperture* flux at bright
sources, but fitting a narrow intrinsic 1.2-pixel Gaussian to the CNN output
severely underestimates blind source flux.

That may indicate a **mismatch between the true effective output source shape
and the Gaussian template used for measurement**. To distinguish measurement
misspecification from lost flux, this independent follow-up calibrates each
saved CNN's effective output source template *away from all test galaxies*.

## Experimental separation

1. For each of six archived CNNs, construct an independent 256x256 synthetic
   calibration field with fixed smooth structure and Gaussian noise, using
   an independent random seed. Insert one known Gaussian source at nominal
   matched-filter SNR 10, use exactly the original Airy PSF nr=5.
2. Reconstruct the *calibration* image with and without the source, using
   common noise. Divide the differential CNN output by the known source
   flux to obtain the empirical output template, including any negative
   side-lobes and changes of width. No held-out galaxy data are used.
3. **Freeze** that model-specific empirical template. On the original six
   held-out galaxy images, choose 12 low-texture source positions per image
   by the exact reference-only grid ranking from the original benchmark.
4. Evaluate three *blind local* flux fitting methods on the simulated image
   with injected sources, no subtraction of source-free control:
   - observed image fitted using the known optical PSF convolved with star;
   - CNN image fitted with a narrow intrinsic Gaussian (previous benchmark);
   - CNN image fitted with the independently calibrated empirical CNN response.
5. Test nominal matched-filter SNR 5, 10 and 20, two new noise seeds, with
   the same background/linear-gradient fit and equal galaxy weighting.

## Predeclared question

Does empirical-response fitting reduce CNN absolute blind flux error by at
least 0.05 (fractional) relative to the uncalibrated Gaussian template on at
least four of six held-out galaxies, particularly at SNR 20?

If yes, the large apparent bias in the previous trial was at least partly
measurement-model mismatch. This *does not* prove unbiased flux conservation:
the empirical response itself may be amplitude-, noise-, position- and
background-dependent. If no, the discrepancy persists despite a modest
independent response calibration and needs further investigation.

This is only a **matched-PSF, Gaussian-noise** follow-up; mismatched PSF,
Poisson noise, phase-aware templates, genuine blind detection and additional
calibrated detectors remain for future work. The original code and upstream
PR are unchanged.
