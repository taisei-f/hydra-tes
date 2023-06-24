import numpy as np
import matplotlib.pyplot as plt
import h5py
import itertools

feature = '../data/feature.hdf5'

###############
# GET DATA
###############
pha = []
arb = 10**8
with h5py.File(feature, 'r') as h5:
    pha.append(h5["Saiteki"]["Pix0"]["PHA"][()]/arb)
    pha.append(h5["Saiteki"]["Pix1"]["PHA"][()]/arb)
    pha.append(h5["Saiteki"]["Pix2"]["PHA"][()]/arb)
    pha.append(h5["Saiteki"]["Pix3"]["PHA"][()]/arb)

###############
# ENERGY CALIBRATION
###############
def response(ka, kb):
    d1 = (5900*kb - 6500*ka)/(5900*6500*(6500-5900))
    d2 = (6500*6500*ka - 5900*5900*kb)/(5900*6500*(6500-5900))
    return d1, d2

search_range = np.array(
    [[105, 120],
     [35, 45],
     [20, 25],
     [12, 16]]
)

Delta_PHA = np.array([10, 4, 2, 1.5])

binnum = np.array(
    [int((search_range[0,1]-search_range[0,0])/(Delta_PHA[0]/60)),
     int((search_range[1,1]-search_range[1,0])/(Delta_PHA[1]/60)),
     int((search_range[2,1]-search_range[2,0])/(Delta_PHA[2]/60)),
     int((search_range[3,1]-search_range[3,0])/(Delta_PHA[3]/60))]
)

energy = []
with h5py.File(feature, 'a') as h5:
    del h5["Calibration"]
    h5.create_group("Calibration")

    for pixnum in range(4):
        h5.create_group("Calibration/Pix"+str(pixnum))
        print("Energy Calibration for Pix" + str(pixnum))
        counts, bins = np.histogram(pha[pixnum], bins=binnum[pixnum], range=(search_range[pixnum,0],search_range[pixnum,1]))
        index_KA = np.argmax(counts)
        index_KB = index_KA+30 + np.argmax(counts[index_KA+30:])

        count_KA = counts[index_KA]
        count_KB = counts[index_KB]
        print("Counts @Ka = %d, @Kb = %d" % (count_KA, count_KB))

        pha_KA = (bins[index_KA]+bins[index_KA+1])/2
        pha_KB = (bins[index_KB]+bins[index_KB+1])/2
        print("PHA @Ka = %.2e, @Kb = %.2e" % (pha_KA, pha_KB))

        d1, d2 = response(pha_KA, pha_KB)
        print("Calib. param : d1 = %.2e, d2 = %.2e" % (d1, d2))

        en = (np.sqrt(4*d1*pha[pixnum]+d2*d2)-d2)/2/d1
        energy.append(en)
        h5.create_dataset(name="Calibration/Pix"+str(pixnum)+"/Energy", data=en)
        print("written in : ~/Calibration/Pix"+str(pixnum)+"/Energy")
        print('')

    h5.create_group("Calibration/All")
    pha_all = list(itertools.chain.from_iterable(pha))
    energy_all = list(itertools.chain.from_iterable(energy))
    h5.create_dataset(name='Calibration/All/PHA', data=pha_all)
    h5.create_dataset(name='Calibration/All/Energy', data=energy_all)
    print("written in : ~/Calibration/All/PHA")
    print("written in : ~/Calibration/All/Energy")
