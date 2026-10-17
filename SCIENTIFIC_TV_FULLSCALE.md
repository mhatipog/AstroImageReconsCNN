# Full-resolution check of the TV validation criterion

Exploratory follow-up to the 256x256 pilot in the separate
`research-tv-selection-audit` branch. We do not alter the original
algorithms, the original M51a noisy observation, the clean FITS reference, or
the saved TV prediction.

This version uses the **original M51a 1024x1024 observation and exactly 500
iterations** in the released `deconv.vc_recons_tv`. It compares three
selected parameters from the **original 20-point logarithmic grid**, two
near the pilot's clean-image optimum and the upper bound.

It measures:
- selection loss against the noisy observation (used in the source code);
- actual clean reference MSE/PSNR (for comparison only);
- pixelwise distance to the original archived TV reconstruction.

The third metric checks whether any tested parameter produces the archived
output. **These are three candidate evaluations, not a full 20-parameter
sweep or reproduction of all six Table I rows**. Conclusions about the
published results require running the full source parameter grid and
investigating any difference with the original author.

Reproduce on CPU using Python 3.11 and NumPy, SciPy, scikit-image, Astropy.
