import numpy as np
import pytest
from adaptive_response_photometry import local_patch, local_response, EPSILON

def test_local_patch_center():
    arr=np.arange(10000,dtype=np.float64).reshape(100,100)
    assert local_patch(arr,(50,50)).shape==(49,49)
    assert local_patch(arr,(50,50))[24,24]==arr[50,50]

def test_eps_validation_without_neural_network():
    with pytest.raises(ValueError):
        local_response(np.zeros((32,32)),np.zeros((32,32)),None,
                       np.ones((32,32)),epsilon=0)
    with pytest.raises(ValueError):
        local_response(np.zeros((32,32)),np.zeros((32,32)),None,
                       np.ones((28,28)),epsilon=EPSILON)
