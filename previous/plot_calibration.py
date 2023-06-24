import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.cm as cm
import matplotlib.colors as color

datafile = '../data/cal_0621.hdf5'
data = []
energy_all = []
pha_all = []
with h5py.File(datafile, 'r') as h5:
    data.append(h5['PHA_hist']['pix1'][()])
    data.append(h5['PHA_hist']['pix2'][()])
    data.append(h5['PHA_hist']['pix3'][()])
    data.append(h5['PHA_hist']['pix4'][()])
    pha_all = h5['Energy']['PHA_all'][()]
    energy_all = h5['Energy']['Energy_all'][()]


##############
# PLOT
##############
# picname = '../data/fig/EnergyCalibration_0621.png'

fig, axes = plt.subplots(
    # nrows=2,
    # ncols=2,
    # squeeze=False,
    tight_layout=True,
    facecolor='whitesmoke'
    )

hist = axes.hist2d(
    energy_all, pha_all,
    bins=50,
    cmap=cm.jet,
    norm=color.LogNorm(),
    )

# axes[0,0].plot(data[0])
# axes[0,0].set_title('pix1')

# axes[0,1].plot(data[1])
# axes[0,1].set_title('pix2')

# axes[1,0].plot(data[2])
# axes[1,0].set_title('pix3')

# axes[1,1].plot(data[3])
# axes[1,1].set_title('pix4')

axes.set_xlabel('Energy (eV)')
axes.set_ylabel('PHA (a.u.)')
axes.grid(linestyle='--')

fig.colorbar(hist[3],ax=axes)
# plt.savefig(picname,bbox_inches='tight',dpi=150)
plt.show()
