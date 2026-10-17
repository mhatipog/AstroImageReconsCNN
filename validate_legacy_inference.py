"""Test faithful forward passes of preserved Keras 1/Theano weights.

No TensorFlow/Keras/Theano dependency: original HDF5 weights, SciPy operations,
and the original saved input/output .mat files provide an independent oracle.

This is a forensic single-patch validation, not a replacement for full inference.
"""
import h5py
import numpy as np
from scipy.io import loadmat
from scipy.signal import correlate2d, convolve2d

PREFIX = "32x32_10x10-64_6x6-16_5x5-1_PSF64_sigma0.010_M51a"
WEIGHTS = f"models/{PREFIX}.h5"
NOISY = "res/none_1024x1024_PSF64_sigma0.010_M51a.mat"
ARCHIVED = "res/cnn_1024x1024_PSF64_sigma0.010_M51a.mat"

def load_weights(path):
    arrays = []
    with h5py.File(path, "r") as file:
        names = [x.decode() if isinstance(x, bytes) else x for x in file.attrs["layer_names"]]
        for name in names:
            group = file[name]
            w = group[f"{name}_W"][:]
            b = group[f"{name}_b"][:]
            arrays.append((w, b))
    assert [x.shape for x, _ in arrays] == [(64, 1, 10, 10), (16, 64, 6, 6), (1, 16, 5, 5)]
    return arrays

def forward(patch, layers, flip):
    x = np.asarray(patch, dtype=np.float32)[None,:,:]
    for kernel, bias in layers:
        out = np.empty((len(bias), x.shape[-2] - kernel.shape[-2] + 1,
                        x.shape[-1] - kernel.shape[-1] + 1), dtype=np.float32)
        for j in range(len(bias)):
            y = np.zeros(out.shape[1:], dtype=np.float32)
            for i in range(x.shape[0]):
                if flip:
                    y += convolve2d(x[i], kernel[j, i], mode="valid")
                else:
                    y += correlate2d(x[i], kernel[j, i], mode="valid")
            out[j] = np.maximum(0, y + bias[j])
        x = out
    assert x.shape == (1,14,14)
    return x[0]

def main():
    weights = load_weights(WEIGHTS)
    input_image = loadmat(NOISY)["Irec"]
    original_pred = loadmat(ARCHIVED)["Irec"]
    assert input_image.shape == original_pred.shape == (1024,1024)
    print("input/output shapes:", input_image.shape)
    results = []
    # Same patch positions as original dsutils.im2patch() with stride=14.
    for flip in [False,True]:
        for ni,nj in [(0,0),(7,13),(25,31)]:
            row,col=ni*14,nj*14
            patch=input_image[row:row+32,col:col+32]
            expected=original_pred[row+9:row+23,col+9:col+23]
            prediction=forward(patch,weights,flip)
            error=prediction-expected
            mae=float(np.mean(np.abs(error)))
            maximum=float(np.max(np.abs(error)))
            rmse=float(np.sqrt(np.mean(error**2)))
            results.append((flip,(ni,nj),mae,maximum,rmse))
            print(f"flip={flip} patch=({ni},{nj}) MAE={mae:.9g} MaxAE={maximum:.9g} RMSE={rmse:.9g}")
    # Only pass if original saved output is numerically accounted for.
    for flip in [False,True]:
        subset=[r for r in results if r[0] == flip]
        if max(r[3] for r in subset)<0.001:
            print("PASS: legacy CNN inference matches archived reconstruction for all sampled patches. flip=",flip)
            return
    raise SystemExit("Not yet matched: investigate convolution semantics and data provenance.")

if __name__ == "__main__":
    main()
