"""Check matched-noise pairing and published PSF normalization."""
import numpy as np
from psf_noise_sensitivity import airy_psf, psnr

def test_original_airy_psf_is_normalized_nonnegative():
    psf = airy_psf(64, 5)
    assert psf.shape == (64, 64)
    assert np.min(psf) >= 0
    np.testing.assert_allclose(np.sum(psf), 1.0, atol=1e-14)
    assert not np.allclose(airy_psf(64, 4), psf)
    assert not np.allclose(airy_psf(64, 6), psf)

def test_psnr_uses_the_original_32_pixel_evaluation_crop():
    truth = np.full((100, 100), 0.5)
    estimate = truth.copy()
    estimate[32:-32,32:-32] += 0.01
    np.testing.assert_allclose(psnr(estimate,truth), 40.0, atol=1e-10)
    estimate[:32] += 100
    np.testing.assert_allclose(psnr(estimate,truth), 40.0, atol=1e-10)
