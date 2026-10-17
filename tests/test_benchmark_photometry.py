import numpy as np
from benchmark_photometry import airy, star, fit_flux, local_template, select_sites, summarize, R

def test_airy_and_star_flux():
    np.testing.assert_allclose(airy(5).sum(),1,rtol=0,atol=1e-14)
    np.testing.assert_allclose(star().sum(),1,rtol=0,atol=1e-14)

def test_source_amplitude_with_planar_nuisance():
    t=np.zeros((2*R+1,2*R+1))
    t[R-5:R+6,R-5:R+6]=star()
    yy,xx=np.mgrid[-R:R+1,-R:R+1]
    image=2.3*t+0.3+0.001*xx-0.002*yy
    # Background-plane removal recovers unbiased flux.
    np.testing.assert_allclose(fit_flux(image,(R,R),t),2.3,atol=1e-12)

def test_site_selection_is_deterministic_spatially_separated():
    image=np.ones((1024,1024))
    a=select_sites(image,20)
    b=select_sites(image,20)
    assert a==b and len(a)==20
    assert len(set(a))==20
    assert min(max(abs(p[0]-q[0]),abs(p[1]-q[1]))
               for i,p in enumerate(a) for q in a[i+1:])>=80
