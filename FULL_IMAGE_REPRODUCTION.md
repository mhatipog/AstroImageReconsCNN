# Reuse the archived 2017 astronomical 3-CNN on modern Python

Rémi Flamary, [Astronomical Image Reconstruction with Convolutional Neural Networks](https://arxiv.org/abs/1612.04526) (EUSIPCO 2017).

This restores **inference** (not training) of the original three-layer Keras
1 / Theano CNN using its unchanged archived HDF5 weights. The implementation
requires CPU PyTorch, NumPy, SciPy and h5py, but **not Theano or Keras**.

## Reproduce the original M51a prediction

Tested on Python 3.11 (Ubuntu/GitHub Actions), with
NumPy 2.4.6, SciPy 1.17.1, h5py 3.16.0 and PyTorch 2.14.1+cpu.

```bash
python -m pip install numpy==2.4.6 scipy==1.17.1 h5py==3.16.0
python -m pip install 'torch==2.14.1+cpu' --index-url https://download.pytorch.org/whl/cpu
python full_image_inference.py --verify
```

The script reads the archived noisy M51a MATLAB image and the original model,
then writes `reproduced/m51a_3cnn_modern.mat`. The optional `--verify`
compares **all 1,048,576 pixels** to the author's archived `cnn` result.

A verified GitHub Actions execution (2026-10-17) measured:
- maximum absolute error: **1.4901161193847656e-6**
- mean absolute error: **5.8961191484740993e-8**
- RMSE: **8.613996232576469e-8**
- number of pixels with absolute error above 1e-5: **zero**

These are **inference parity** metrics, not a new accuracy/PSNR result.
[Original full-image numerical validation](https://github.com/mhatipog/AstroImageReconsCNN/actions/runs/39022995042).

## Apply the archived network to another degraded image

```bash
python full_image_inference.py --input your_degraded_image.npy \
    --weights models/32x32_10x10-64_6x6-16_5x5-1_PSF64_sigma0.010_M51a.h5 \
    --output your_reconstruction.mat
```

Inputs can be 2D NumPy arrays (`.npy`) or MATLAB `.mat` files containing
`Irec`. FITS inputs are also supported if `astropy` is separately
installed. New inputs must already be scaled and degraded comparably to the
training images; performance on other domains is **not established**.

The original `dsutils.py` predicts overlapping 32x32 input patches, yielding
14x14 output patches written into the middle of the frame. This version uses
larger tiled blocks with the same 18-pixel receptive-field halo, patch
alignment and zero-valued outer border. Keras/Theano's historical mathematical
convolution convention requires flipping the learned spatial kernels before
PyTorch `conv2d`. The tests independently verify tile equivalence.

**Scope:** This implementation restores the archived three-layer 3-CNN model.
It does not retrain any network or restore the single-layer model's missing
weights. Credit and citation remain with the original author and paper.
