import numpy as np
import matplotlib.pyplot as plt
import h5py
from module import readhdf5

r = readhdf5.ReadHDF5('../../data/run041pn.hdf5')
r.OpenHDF5()

t = r.time
p = r.pulse
n = r.noise

tau = []

def Mask(pulse):
    a = pulse<line_20
    b = pulse>line_80
    return a & b

for event_num in range(len(p)):
    baseline = np.average(n[event_num])
    p_max = np.min(p[event_num])
    ph_index = np.argmin(p[event_num])
    ph = baseline - p_max
    line_20 = baseline - 0.2*ph
    line_80 = baseline - 0.8*ph
    
    first_half = t[:ph_index]
    first_sliced = first_half[Mask(pulse=p[event_num][:ph_index])]
    
    second_half = t[ph_index:]
    second_sliced = second_half[Mask(pulse=p[event_num][ph_index:])]

    if first_sliced.size != 0 and second_sliced.size != 0:
        tau.append([first_sliced[-1]-first_sliced[0], second_sliced[-1]-second_sliced[0]])
        # print(first_sliced[-1]-first_sliced[0], second_sliced[-1]-second_sliced[0])
    else:
        tau.append([np.nan, np.nan])

tau_reshape = np.array(tau).reshape([-1,2])
print(tau_reshape)

with h5py.File('../../data/feature.hdf5', 'w') as h5:
    h5.create_dataset('tau', data=tau_reshape)

r.CloseHDF5()