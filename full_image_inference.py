"""Run the original Keras-1/Theano 3-CNN on a full image with modern PyTorch.

No legacy deep-learning environment is needed. The archived Keras weight arrays
are loaded directly from HDF5. Theano performed mathematical convolution;
PyTorch conv2d performs cross-correlation, so flip both spatial filter axes.

The original dsutils.py splits a 1024x1024 image into overlapping 32x32 patches,
with a 14x14 output grid per patch and a 9-pixel outer crop. Evaluating a large
tile with the exact same 18-pixel receptive-field halo is mathematically
equivalent to processing each 32x32 patch independently and much faster.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import h5py
from scipy.io import loadmat, savemat
import torch
import torch.nn.functional as F

MODEL = Path("models/32x32_10x10-64_6x6-16_5x5-1_PSF64_sigma0.010_M51a.h5")
INPUT = Path("res/none_1024x1024_PSF64_sigma0.010_M51a.mat")
REFERENCE = Path("res/cnn_1024x1024_PSF64_sigma0.010_M51a.mat")
OUTPUT = Path("reproduced/m51a_3cnn_modern.mat")
PATCH_INPUT = 32
PATCH_OUTPUT = 14
HALO = PATCH_INPUT - PATCH_OUTPUT  # 18 pixels total, 9 on each side


def _load_array(path: Path, key: str) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".mat":
        array = loadmat(path)[key]
    elif ext == ".npy":
        array = np.load(path, allow_pickle=False)
    elif ext in (".fit", ".fits"):
        try:
            from astropy.io.fits import getdata
        except ImportError as exc:
            raise RuntimeError("Install astropy to read FITS inputs") from exc
        array = getdata(path)
    else:
        raise ValueError(f"Unsupported image extension: {ext}")
    result = np.asarray(array, dtype=np.float32)
    if result.ndim != 2 or min(result.shape) < PATCH_INPUT:
        raise ValueError(f"Expected a 2D image at least 32x32, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Input image contains NaN or infinity")
    return result


def load_layers(path: Path) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    layers = []
    with h5py.File(path, "r") as file:
        names = [n.decode() if isinstance(n, bytes) else str(n)
                 for n in file.attrs["layer_names"]]
        for name in names:
            weights = np.asarray(file[name][f"{name}_W"], dtype=np.float32)
            bias = np.asarray(file[name][f"{name}_b"], dtype=np.float32)
            if weights.ndim != 4 or bias.ndim != 1:
                raise ValueError(f"Invalid convolution layer: {name}")
            # Old Theano conv: mathematical convolution.
            # PyTorch F.conv2d: cross-correlation.
            # Flip each 2D kernel to preserve the trained computation.
            weight = torch.from_numpy(weights.copy()).flip(-1, -2).contiguous()
            layers.append((weight, torch.from_numpy(bias.copy())))
    expected = [(64, 1, 10, 10), (16, 64, 6, 6), (1, 16, 5, 5)]
    if [tuple(w.shape) for w, _ in layers] != expected:
        raise ValueError("This script requires the archived three-layer 3-CNN weights")
    return tuple(layers)


@torch.no_grad()
def forward(image: torch.Tensor, layers) -> torch.Tensor:
    x = image
    for weights, bias in layers:
        x = F.relu(F.conv2d(x, weights, bias=bias, stride=1, padding=0))
    return x


def reconstruct(input_image: np.ndarray, layers, tile_size: int = 196) -> np.ndarray:
    """Return the reconstructed image on the original Keras output grid."""
    if tile_size <= 0 or tile_size % PATCH_OUTPUT:
        raise ValueError("Tile size must be a positive multiple of 14")
    height, width = input_image.shape
    nrows = (height - HALO) // PATCH_OUTPUT
    ncols = (width - HALO) // PATCH_OUTPUT
    if nrows < 1 or ncols < 1:
        raise ValueError("Input too small for the archived model")
    extent_h, extent_w = nrows * PATCH_OUTPUT, ncols * PATCH_OUTPUT
    # dsutils.patch2im() initializes a zero image and writes only the tiled
    # reconstruction into the interior, leaving the 9-pixel halo untouched.
    result = np.zeros((height, width), dtype=np.float32)
    margin = HALO // 2
    for top in range(0, extent_h, tile_size):
        block_h = min(tile_size, extent_h - top)
        for left in range(0, extent_w, tile_size):
            block_w = min(tile_size, extent_w - left)
            patch = np.ascontiguousarray(
                input_image[top:top + block_h + HALO,
                            left:left + block_w + HALO], dtype=np.float32
            )
            tensor = torch.from_numpy(patch)[None, None]
            y = forward(tensor, layers)[0, 0].cpu().numpy()
            if y.shape != (block_h, block_w):
                raise AssertionError(f"Unexpected model output {y.shape}")
            result[margin + top:margin + top + block_h,
                   margin + left:margin + left + block_w] = y
    return result


def verify(prediction: np.ndarray, archived: np.ndarray, max_tolerance: float) -> dict:
    if archived.shape != prediction.shape:
        raise ValueError(f"Reference shape {archived.shape} != output {prediction.shape}")
    error = np.abs(prediction.astype(np.float64) - archived.astype(np.float64))
    stats = {
        "mae": float(error.mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "max_error": float(error.max()),
        "pixels": int(error.size),
        "above_1e-5": int(np.count_nonzero(error > 1e-5)),
    }
    print("Original vs restored full-image inference:", stats, flush=True)
    if not np.all(np.isfinite(error)) or stats["max_error"] > max_tolerance:
        raise AssertionError(
            f"Full-image inference does not reproduce archive to {max_tolerance:g}"
        )
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--input-key", default="Irec")
    parser.add_argument("--weights", type=Path, default=MODEL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--tile-size", type=int, default=196)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--verify", action="store_true",
                        help="Compare all pixels against archived M51a reconstruction")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--max-error", type=float, default=1e-4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    data = _load_array(args.input, args.input_key)
    print(f"Input: {data.shape}; loading archived trained weights", flush=True)
    result = reconstruct(data, load_layers(args.weights), tile_size=args.tile_size)
    if args.verify:
        original = _load_array(args.reference, "Irec")
        verify(result, original, args.max_error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    savemat(args.output, {"Irec": result}, do_compression=True)
    print(f"Wrote restored prediction: {args.output}", flush=True)


if __name__ == "__main__":
    main()
