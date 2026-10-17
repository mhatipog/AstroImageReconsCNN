## Full-image model recovery (CPU)

The independent three-patch verification described above now extends to full-image
inference. The archived 2016 Keras 1 network is recreated with present-day
PyTorch convolution operators and the *original* HDF5 learned weights.

To run on the original archived noisy M51a observation:

```sh
python -m pip install 'numpy>=1.26,<3' 'scipy>=1.11,<2' 'h5py>=3.10,<4'
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python full_image_inference.py --verify
```

The tool reprocesses every pixel covered by the original patch tiling,
initializes the same zero-valued outer border, and reports maximum error,
MAE and RMSE against the author's stored `res/cnn_..._M51a.mat`.
The output `reproduced/m51a_3cnn_modern.mat` is generated, not committed.

The original trained model can also process a different *degraded* 2D image:

```sh
python full_image_inference.py --input your_input.npy --output your_output.mat
```

The expected input is an already blurred/noisy float image scaled like the
original training data. New-image use is a demonstration of executable
archived inference, **not** a claim of generalization or new astronomical
accuracy. The inference script only restores the archived three-layer
M51a network; it does not retrain the model or restore the unarchived
single-layer CNN weights.

This implementation evaluates larger aligned output tiles instead of 32×32
patches individually. The 18-pixel receptive-field halo and 14-pixel
output stride preserve the original `dsutils.py` patch placement and border.
