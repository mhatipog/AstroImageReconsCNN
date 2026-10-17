"""Minimal synthetic checks of full-image inference tiling semantics."""
import numpy as np
import torch
from full_image_inference import reconstruct, forward, HALO


def test_full_image_matches_dense_valid_conv_and_zero_border():
    gen = np.random.default_rng(777)
    shapes = [(64, 1, 10, 10), (16, 64, 6, 6), (1, 16, 5, 5)]
    layers = []
    for shape in shapes:
        weights = torch.from_numpy((gen.normal(size=shape) * 0.003).astype("float32"))
        bias = torch.zeros(shape[0], dtype=torch.float32)
        layers.append((weights, bias))
    image = gen.normal(size=(88, 102)).astype("float32")
    predicted = reconstruct(image, layers, tile_size=28)
    dense = forward(torch.from_numpy(image)[None, None], layers)[0, 0].numpy()
    nh = (image.shape[0] - HALO) // 14 * 14
    nw = (image.shape[1] - HALO) // 14 * 14
    expected = np.zeros_like(predicted)
    expected[9:9 + nh, 9:9 + nw] = dense[:nh, :nw]
    np.testing.assert_allclose(predicted, expected, atol=1e-5, rtol=1e-5)
    assert np.count_nonzero(predicted[:9]) == 0
    assert np.count_nonzero(predicted[:, :9]) == 0
