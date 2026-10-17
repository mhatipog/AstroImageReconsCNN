# Audit of the original Prox-TV selection criterion

In Flamary's original `run_deconv_all.py`, the Wiener regularization
parameter is selected against `I0` (the clean test image), whereas
Prox-TV is selected by minimizing `mse_crop(I, vc_recons_tv(I, ...))`.
Here `I` is the degraded observation, **not** the clean image.

That is a verifiable difference in how the source selects parameters;
it does **not**, by itself, show that the published Table I is wrong.

The script `audit_tv_selection.py` investigates whether the distinction
changes selected parameters or clean-image PSNR on three archived galaxies.

To keep this a bounded exploratory experiment, it uses central
**256x256** crops, 12 logarithmically spaced parameter candidates
(`mu=1..10000`) and **100** iterations of the *original*, unmodified
`deconv.vc_recons_tv`. The original released script uses 20 candidates
and 500 iterations. The paper refers to 100 iterations.

The procedure measures both the original criterion (distance to degraded
input) and the clean-reference criterion on each returned reconstruction.
It records all losses and the PSNR penalty of selecting with the degraded
observation. All runs are CPU-only.

**Do not present this pilot as a correction to the published 1024x1024
table.** Boundary conditions, selection grids and iteration counts differ
from the archived full-scale comparison. A consequential result warrants
a full-scale controlled replication and communication with the author
*before* any upstream issue or assertion of a publication error.

Reproduce on Python 3.11:

```sh
python -m pip install numpy==2.4.6 scipy==1.17.1 h5py==3.16.0 \
    astropy==7.1.0 scikit-image==0.25.2
python audit_tv_selection.py
```
