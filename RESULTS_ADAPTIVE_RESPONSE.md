# Adaptive source-response photometry — completed pilot

**Status:** exploratory six-galaxy, synthetic-observation study; not a
publication or an upstream claim. Executed 17 October 2026.

Original article and pretrained six leave-one-galaxy-out networks:
Rémi Flamary, *Astronomical Image Reconstruction with Convolutional Neural
Networks*, EUSIPCO 2017.

## Question and predeclared screen

Can a **local numerical directional response** of the pretrained CNN,
measured by a fixed epsilon=0.05 artificial perturbation **at the
source-present observed image**, reduce known-position photometric error
versus the independent, separately calibrated fixed CNN response template?

The stopping rule written before running this experiment was an improvement
in mean absolute *fractional* flux error >=0.03 in >=4 of six galaxies
at a specified SNR. This is an exploratory screen, **not a significance
test**.

## Completed protocol

- The **six** unchanged original leave-one-galaxy-out models.
- Original simulated Airy PSF `nr=5`, Gaussian noise sigma=0.01.
- **SNR 5, 10, 20**, two fresh random noise seeds (333/444), 12
  known artificial-source positions per held-out galaxy.
- Four *source-present, known-coordinate* photometric estimators:
  correct-PSF fitting on the observed image, intrinsic-Gaussian
  fitting on the CNN image, independent empirical-template fitting,
  and observation-adaptive finite-response fitting.
- 144 measurements per method/SNR, **1,728 total**.
- No subtraction of a source-absent test output was used for the fitted
  photometry. The *local directional derivative* uses a new known
  perturbation to the observed input, not the original test label.
- The six galaxies—not repeated sites—are the analysis clusters.

## Results

Mean absolute fractional flux error, equal galaxy weighting
(smaller is better):

| Nominal SNR | Direct PSF fit | CNN intrinsic template | CNN fixed independent template | CNN adaptive response | Adaptive wins (of 6) |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.9914 | 0.7105 | 0.7037 | **0.6313** | **5** |
| 10 | 0.4786 | 0.5964 | **0.3301** | 0.3355 | 3 |
| 20 | 0.2346 | 0.5045 | 0.2012 | **0.1966** | 3 |

Galaxy-level differences (adaptive minus fixed-template absolute flux error),
in order M31, Hoag, M51a, M81, M101, M104:

- SNR 5: **[-0.0223, -0.0663, -0.1968, +0.0507, -0.0954, -0.1037]**
- SNR 10: [-0.0450, +0.1085, -0.1021, +0.0935, -0.0296, +0.0071]
- SNR 20: [-0.0193, +0.0596, -0.0609, +0.0598, -0.0717, +0.0048]

No degeneracy of the local adaptive templates occurred in this run.

## Decision

**Passes the exploratory screen at SNR 5 only.** Error falls 0.7037 -> 0.6313
(absolute improvement 0.0723), in five of six galaxies. At SNR 10
it is marginally worse, and at SNR 20 its improvement is very small,
with only three of six galaxies improved in both cases.

Do **not** replace independent calibration generally with adaptive
finite differences. The extra CNN reconstruction and dependence on
perturbation size are not justified at moderate/high SNR by these data.

## Scientific limitations

- Every galaxy comes from the same six-image historical experiment.
  New noise realizations do not create a genuinely independent sky population.
- Forced photometry **knows source coordinates**. No blind detection
  completeness, false-discovery, or survey claims follow.
- The PSF and intrinsic star shape are given by the synthetic generator;
  not estimated from a real telescope.
- The adaptive template is a derivative around a *source-present*
  observation. Nonlinear effects could make a secant or multi-amplitude
  response more appropriate. Epsilon 0.05 was fixed, not optimized.
- The apparent improvements could partly reflect background fit effects
  and source profile misspecification, rather than universal flux preservation.
- The directly fitted observed image has large errors at low SNR in these
  structured archival backgrounds. This is not evidence that the restored
  network is superior to a fully calibrated instrument photometry pipeline.
- Only six independent image clusters, exploratory comparisons, no
  multiplicity-adjusted statistical inference.

**Scientifically justified follow-up:** test whether the faint-SNR advantage
survives spatially independent artificial skies with known catalogs and
*blinded* detection; compare to properly calibrated modern methods. Do not
contact the original author about this preliminary result.

## Verification

[Successful GitHub Actions run](https://github.com/mhatipog/AstroImageReconsCNN/actions/runs/39030257633)
— 2 unit tests passed; full science experiment passed; downloadable
`adaptive_response_trials.csv` and `adaptive_response_summary.csv`.
