"""Simple unit checks for artificial-source aperture accounting."""
import numpy as np
from point_source_response import aperture, injected_field

def test_injected_total_flux_is_known():
    positions=[(120,120),(280,280)]
    field=injected_field((2200,2200),positions)
    np.testing.assert_allclose(field.sum(),2.0,atol=1e-12)
    local=field[1200:2200,1200:2200]
    for y,x in positions:
        np.testing.assert_allclose(aperture(local,y,x,8),1.0,atol=1e-10)
