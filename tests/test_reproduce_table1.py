"""Fast schema and numerical checks using synthetic MATLAB data."""
import numpy as np
from scipy.io import savemat
from reproduce_table1 import (IMAGES, METHODS, PUBLISHED_PSNR,
                              PUBLISHED_MEAN_PSNR, differences, psnr, recover)

def test_psnr_aggregation_matches_paper_definition():
    mse = 10 ** (-PUBLISHED_PSNR / 10)
    actual = psnr(mse.mean(axis=0))
    np.testing.assert_allclose(actual, PUBLISHED_MEAN_PSNR, atol=0.005)
    assert np.max(np.abs(PUBLISHED_PSNR.mean(axis=0) - actual)) > 0.1

def test_loading_original_matlab_schema(tmp_path):
    mse = 10 ** (-PUBLISHED_PSNR / 10)
    for i, method in enumerate(METHODS):
        savemat(tmp_path / f'{method}_1024x1024_PSF64_sigma0.010_all.mat', {
            'mse': mse[:, i][None, :],
            't': np.ones((1, len(IMAGES))),
        })
    loaded, times = recover(tmp_path)
    np.testing.assert_allclose(loaded, mse)
    actual = differences(loaded, times)
    np.testing.assert_allclose(actual[0], PUBLISHED_PSNR, atol=1e-12)
    assert actual[3] < 0.005
    assert actual[4] < 0.005

def test_invalid_mse_fails():
    for x in [0.0, -1.0, np.nan, np.inf]:
        try:
            psnr([x])
        except ValueError:
            pass
        else:
            raise AssertionError(f'{x} did not fail')
