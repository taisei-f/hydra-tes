import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as color
import h5py
import itertools

file = '../data/PHA.hdf5'
arb = 10**10
output = '../data/Energy_0621.hdf5'

pix = [4, 1, 2, 3]

###############
# GET DATA
###############
pha = []
with h5py.File(file, 'r') as h5:
    pha.append(h5['PHA']['pix0'][()]/arb)
    pha.append(h5['PHA']['pix1'][()]/arb)
    pha.append(h5['PHA']['pix2'][()]/arb)
    pha.append(h5['PHA']['pix3'][()]/arb)



###############
# ENERGY CALIBRATION
###############
def response(ka, kb):
    d1 = (5900*kb - 6500*ka)/(5900*6500*(6500-5900))
    d2 = (6500*6500*ka - 5900*5900*kb)/(5900*6500*(6500-5900))
    return d1, d2


energy = []
with h5py.File(output, 'a') as h5:
    # h5.create_group('Energy')

    for pixnum in range(4):
        print("Energy Calibration for pix. %d" % pix[pixnum])
        n, bins = np.histogram(pha[pixnum], bins=100)
        index_KA = np.argmax(n)
        index_KB = index_KA+3 + np.argmax(n[index_KA+3:])

        count_KA = n[index_KA]
        count_KB = n[index_KB]
        print("counts @Ka = %d, @Kb = %d" % (count_KA, count_KB))

        pha_KA = (bins[index_KA]+bins[index_KA+1])/2
        pha_KB = (bins[index_KB]+bins[index_KB+1])/2
        print("PHA @Ka = %.2e, @Kb = %.2e" % (pha_KA, pha_KB))

        d1, d2 = response(pha_KA, pha_KB)
        print("Calib. param : d1 = %.2e, d2 = %.2e" % (d1, d2))
        print('')

        energy.append((np.sqrt(4*d1*pha[pixnum]+d2*d2)-d2)/2/d1)

    # h5.create_dataset(name='Energy/pix4', data=energy[0])
    # h5.create_dataset(name='Energy/pix1', data=energy[1])
    # h5.create_dataset(name='Energy/pix2', data=energy[2])
    # h5.create_dataset(name='Energy/pix3', data=energy[3])

    pha_all = list(itertools.chain.from_iterable(pha))
    energy_all = list(itertools.chain.from_iterable(energy))
    # h5.create_dataset(name='Energy/PHA_all', data=pha_all)
    # h5.create_dataset(name='Energy/Energy_all', data=energy_all)


###############
# PLOT
###############
picname = '../data/fig/EnergyCalibration_0620.png'

fig, ax = plt.subplots(
    # nrows=2,
    # ncols=2,
    # squeeze=False,
    tight_layout=True,
    facecolor='whitesmoke'
    )

hist = ax.hist2d(
    energy_all, pha_all,
    bins=50,
    cmap=cm.jet,
    norm=color.LogNorm(),
    )

# ax.set_title("pix."+str(pix[pixnum]))
ax.set_xlabel('Energy (keV)')
ax.set_ylabel('PHA (a.u.)')
ax.grid(linestyle='--')

fig.colorbar(hist[3],ax=ax)
# plt.savefig(picname,bbox_inches='tight',dpi=150)
