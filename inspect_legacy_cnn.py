"""Inspect the saved Keras 1 weights without loading Keras/Theano."""
import h5py
from pathlib import Path

path=Path("models/32x32_10x10-64_6x6-16_5x5-1_PSF64_sigma0.010_M51a.h5")
with h5py.File(path, "r") as f:
    print("HDF5 root attributes:", {k:str(v) for k,v in f.attrs.items()})
    def dump(name, obj):
        if isinstance(obj,h5py.Dataset):
            print("DATA",name,"shape=",obj.shape,"dtype=",obj.dtype)
        else:
            print("GROUP",name,"attrs=",{k:str(v) for k,v in obj.attrs.items()})
    f.visititems(dump)
