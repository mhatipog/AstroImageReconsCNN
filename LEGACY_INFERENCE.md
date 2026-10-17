# Independent verification of the archived 3-CNN model

This experiment checks the exact numerical forward pass of Rémi Flamary's
[Astronomical Image Reconstruction with Convolutional Neural Networks](https://arxiv.org/abs/1612.04526)
(EUSIPCO 2017). It uses only the **original archived weights, a saved noisy
input, and the author's saved CNN reconstruction**.

## Run

```bash
python -m pip install 'numpy>=1.26,<3' 'scipy>=1.11,<2' 'h5py>=3.10,<4'
python validate_legacy_inference.py
```

No Theano, Keras, TensorFlow, pretrained-weight conversion, or retraining.

The script reads the archived M51a 3-CNN HDF5 model, the noisy M51a
image in `res/none_..._M51a.mat` and the original prediction in
`res/cnn_..._M51a.mat`. It computes three deterministic 32×32 input patches
at different image positions and compares the resulting 14×14 predictions
against exactly the corresponding regions in the archived output image.

The 2016 Keras/Theano convolution convention is important here: **the HDF5
filters must be applied with mathematical convolution**, not unflipped
cross-correlation. Both possibilities are tested, so the wrong one acts as a
useful negative control.

At the time of validation, the correct convention agreed with the saved
image to a maximum absolute pixel error of `2.23517418e-7` across three
sampled patches. The unflipped interpretation deviated by as much as
`1.90667212e-2` on those same samples.

## What this verifies

- The archived 3-CNN layer dimensions, filter tensors, biases and ReLU
  activations permit an independent, numerically faithful forward pass.
- The saved convolution weights are recoverable in current Python/SciPy.
- The patch-to-image pixel mapping matches the original `dsutils.py`.

## What it does not verify

- Full-image inference speed, training reproducibility, or results on new
  astronomical data.
- The 1-CNN model (its trained weight files are not in this archive).
- Any downstream data or results beyond the three sampled M51a patches.

The original author and publication remain the primary scientific sources.
This is a limited independent archival verification, not a new claim about
CNN performance.
