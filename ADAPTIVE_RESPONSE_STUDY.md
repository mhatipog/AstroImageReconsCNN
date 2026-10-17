# Local empirical response of the archived CNN

An **exploratory follow-up** to the independent calibration in
`research-effective-response-calibration`. This does not modify any
2017 original method or the upstream Figure 3 PR.

## Motivating observation

One fixed CNN response template obtained on an independently generated
synthetic scene improved photometry on held-out galaxies, but its
performance varied with source brightness and between galaxies. The
restoration network contains nonlinear ReLU activations. Its *effective
point-source response* need not be the intrinsic Gaussian star or the
same across input backgrounds.

## Experiment

For a known trial coordinate and observation `I`, the model output is
`f(I)`. Let `P` be the blurred unit-flux artificial-star input.
The local response is approximated using a **finite directional difference**:

```text
T_local(I) = [ f(I + epsilon * P) - f(I) ] / epsilon
```

The perturbation flux is **fixed at epsilon=0.05** in the same normalized
image-flux units for every experiment. Perturbations at multiple separated
known coordinates are computed together; the CNN has a local receptive
field so the local 49×49 region contains only the corresponding source
response. These templates are computed at the **observed source-present
image**; no source-absent control or source flux label is used to compute
them. This is not full Bayesian inference, and the derivative at
source-present input is not necessarily the integrated response from
source-absent input to source-present input.

The response is used to fit the unknown source amplitude and a nuisance
tilted background in the **source-present restored image alone**.
We compare on exactly the same held-out trial coordinates:

1. Correct-PSF photometry on the degraded observation;
2. CNN with an uncalibrated intrinsic-Gaussian source template;
3. CNN with a **frozen, independently calibrated** synthetic-source
   response (prior control);
4. CNN with an **observation-adaptive finite-difference** source response.

Six original leave-one-galaxy-out networks, each on its held-out galaxy,
PSF nr=5, Gaussian noise sigma=0.01, SNR 5/10/20 and two new noise
realizations (333 and 444). Twelve predetermined reference-selected sites
per galaxy. Each galaxy counts as one independent cluster.

**Decision rule:** Adaptive calibration is considered promising only if it
reduces mean absolute fractional flux error by at least 0.03 compared
with the independent template in at least four of six galaxies at a
specified SNR. Otherwise this added computation is not justified. This
threshold is exploratory, not a statistical significance criterion.

Important limitations:
- Trial locations are known (forced photometry); no blind source finder.
- The method assumes the correct simulated optical PSF and source profile.
- The galaxies and insertion sites were used in preceding explorations;
  the new noise seeds do not create an independent astronomical dataset.
- Directional differences are evaluated at the source-containing image
  and may be nonlinear in the chosen perturbation size.
- This tests a generic measurement technique, not a new restoration CNN.
- Unlike a static template, this strategy needs an additional CNN
  reconstruction per observed frame containing the injection sites.
- No outcome-derived tuning of epsilon or templates is performed.

Run via the included GitHub Actions workflow with pinned Python 3.11,
NumPy/SciPy/h5py, scikit-image, Astropy and CPU PyTorch. Outputs are
`results/adaptive_response_trials.csv` and
`results/adaptive_response_summary.csv`.

This is research-only. It should **not** be sent to the original author
until it has survived independent test data and scientific review.
