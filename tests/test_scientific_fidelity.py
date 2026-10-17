"""Independent small synthetic checks; no telescope data needed."""
import numpy as np
from scientific_fidelity_benchmark import (airy, choose_sites, star_kernel, make_sources,
      translated_psf_template, nominal_amplitudes, fit_flux_local,
      sample_noise, summarize, PSF_RADII, NOISE_MODELS, GALAXIES, SNR_LEVELS)


def test_psf_star_and_disjoint_preselected_positions():
    assert np.isclose(airy(5).sum(),1)
    assert np.isclose(star_kernel().sum(),1)
    assert not np.allclose(airy(4),airy(5))
    positions, cal, null = choose_sites(np.ones((1024,1024))*.2)
    assert len(positions)==len(cal)==len(null)==12
    assert len(set(positions+cal+null))==36
    assert np.isclose(make_sources((1024,1024),positions,[2]*12).sum(),24)


def test_nominal_matched_filter_snr_and_observation_recovery():
    psf=airy(5)
    template=translated_psf_template(psf)
    expected_snr=5
    background=np.full((128,128),.2)
    site=[(64,64)]
    flux=nominal_amplitudes(background,site,template,'gaussian',expected_snr)[0]
    observed_snr=flux * np.sqrt(np.sum(template**2/0.01**2))
    np.testing.assert_allclose(observed_snr,expected_snr,rtol=1e-12)
    # On an exactly flat background, properly centered PSF-fit recovers flux.
    from scipy.signal import fftconvolve
    obs=background + flux*fftconvolve(make_sources((128,128),site,[1.]),psf,mode='same')
    measured=fit_flux_local(obs,site[0],template)
    np.testing.assert_allclose(measured,flux,atol=1e-8)


def test_photon_superposition_follows_expected_mean():
    psf=airy(5)
    base=np.full((128,128),0.2)
    added=fft_delta=np.maximum(0, np.asarray([[0]]))  # make an actual source below
    from scipy.signal import fftconvolve
    added=fftconvolve(make_sources((128,128),[(64,64)],[0.7]),psf,'same')
    sums=[]
    for i in range(60):
        first,second=sample_noise(base,added,'photon_read',np.random.default_rng(1000+i))
        sums.append(float((second-first).sum()))
    assert abs(np.mean(sums)-.7)<.1