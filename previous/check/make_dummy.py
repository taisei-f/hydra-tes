import numpy as np
import h5py
import random

file = '../data/Dummy.hdf5'

def pulse(a=1.0, t1=1e-3, t2=5e-3, dlen=1024, dt=1e-4):
    t = np.linspace(0, dt*(dlen-1), dlen)
    y = a * (np.exp(-t/t2) - np.exp(-t/t1)) + random.gauss(mu=0.0, sigma=)
    return t, y

with h5py.File(file, 'w') as h5:
    