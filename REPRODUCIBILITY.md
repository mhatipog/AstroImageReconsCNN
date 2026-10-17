# Reproducing published Table I without retraining

Reference: Rémi Flamary, *Astronomical Image Reconstruction with Convolutional Neural Networks*, EUSIPCO 2017. [Paper](https://doi.org/10.23919/EUSIPCO.2017.8081654).

The original `res/` directory contains MATLAB numerical results for six astronomical images and five methods. This workflow checks the **archived numerical outputs**, not a new execution of training or deconvolution. No GPU, Theano, or Keras is needed.

On a clone:

```sh
python -m pip install 'numpy>=1.26,<3' 'scipy>=1.11,<2' pytest
python -m pytest -q tests/test_reproduce_table1.py
python reproduce_table1.py
```

The CSV is written to `reproduced/table1.csv`. The command fails when saved results differ from Table I beyond its published two-decimal precision. Inputs in `res/` are read-only.

**PSNR detail:** The original paper's Table I script computes aggregate PSNR as `-10 log10(mean(MSE))`, not `mean(-10 log10(MSE))`.

**Limitations:** Archived timings refer to the author's original hardware. This only recovers historical results, not modern runtime or CNN predictive performance. Cite Flamary (2017); the original repository is MIT-licensed.

## Re-rendering Figure 3

```sh
python -m pip install 'matplotlib>=3.8,<4' 'astropy>=6,<9'
python reproduce_figure3.py
```

This reads the archived M51a FITS image and each original `res/*_M51a.mat` file. It asserts that 1-CNN and 3-CNN are numerically distinct and writes `reproduced/figure3.png`. This is a new presentation of the original saved images with a consistent grayscale range; it is not a retrained result and is not pixel-identical to the published figure layout. In particular, the original script accidentally plotted the 1-CNN array in both CNN panels.
