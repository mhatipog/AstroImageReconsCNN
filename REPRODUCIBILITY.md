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
