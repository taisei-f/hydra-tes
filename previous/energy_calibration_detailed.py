import numpy as np
import matplotlib.pyplot as plt
import h5py
import itertools

phafile = '../data/PHA.hdf5'
output = '../data/cal_0621.hdf5'


###############
# GET DATA
###############
data = []
arb = 10**10
with h5py.File(phafile, 'r') as h5:
    data.append(h5['PHA']['pix0'][()]/arb)
    data.append(h5['PHA']['pix1'][()]/arb)
    data.append(h5['PHA']['pix2'][()]/arb)
    data.append(h5['PHA']['pix3'][()]/arb)

# swap pix number
pha = [data[1], data[2], data[3], data[0]]

###############
# ENERGY CALIBRATION
###############
def response(ka, kb):
    d1 = (5900*kb - 6500*ka)/(5900*6500*(6500-5900))
    d2 = (6500*6500*ka - 5900*5900*kb)/(5900*6500*(6500-5900))
    return d1, d2

search_range = np.array(
    [[1.2, 1.5], # pix1
     [1.0, 1.3], # pix2
     [0.7, 1.0], # pix3
     [0.10, 0.17]]) # pix4

binnum = np.array(
    [int((search_range[0,1]-search_range[0,0])*1e3),
     int((search_range[1,1]-search_range[1,0])*1e3),
     int((search_range[2,1]-search_range[2,0])*1e3),
     int((search_range[3,1]-search_range[3,0])*1e3)])

# pha/1bin * ??? = 0.05 or 0.005
a = np.array(
    [int(0.05/(search_range[0,1]-search_range[0,0])*binnum[0]),
     int(0.05/(search_range[1,1]-search_range[1,0])*binnum[1]),
     int(0.05/(search_range[2,1]-search_range[2,0])*binnum[2]),
     int(0.005/(search_range[3,1]-search_range[3,0])*binnum[3])]
)

energy = []
with h5py.File(output, 'a') as h5:
    h5.create_group('PHA_hist')
    h5.create_group('Energy')

    for pixnum in range(4):
        print("Energy Calibration for pix" + str(pixnum+1))
        counts, bins = np.histogram(pha[pixnum], bins=binnum[pixnum], range=(search_range[pixnum,0],search_range[pixnum,1]))
        index_KA = np.argmax(counts)
        index_KB = index_KA+a[pixnum] + np.argmax(counts[index_KA+a[pixnum]:])
        h5.create_dataset(name='PHA_hist/pix'+str(pixnum+1), data=counts)

        count_KA = counts[index_KA]
        count_KB = counts[index_KB]
        print("counts @Ka = %d, @Kb = %d" % (count_KA, count_KB))

        pha_KA = (bins[index_KA]+bins[index_KA+1])/2
        pha_KB = (bins[index_KB]+bins[index_KB+1])/2
        print("PHA @Ka = %.2e, @Kb = %.2e" % (pha_KA, pha_KB))

        d1, d2 = response(pha_KA, pha_KB)
        print("Calib. param : d1 = %.2e, d2 = %.2e" % (d1, d2))
        print('')

        en = (np.sqrt(4*d1*pha[pixnum]+d2*d2)-d2)/2/d1
        energy.append(en)
        h5.create_dataset(name='Energy/pix'+str(pixnum+1), data=en)

    pha_all = list(itertools.chain.from_iterable(pha))
    energy_all = list(itertools.chain.from_iterable(energy))
    h5.create_dataset(name='Energy/PHA_all', data=pha_all)
    h5.create_dataset(name='Energy/Energy_all', data=energy_all)
