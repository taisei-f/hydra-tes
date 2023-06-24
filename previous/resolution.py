import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

datafile = '../data/cal_0621.hdf5'

energy = []
with h5py.File(datafile, 'r') as h5:
    energy.append(h5['Energy']['pix1'][()])
    energy.append(h5['Energy']['pix2'][()])
    energy.append(h5['Energy']['pix3'][()])
    energy.append(h5['Energy']['pix4'][()])


###############
# FIT & PLOT
###############
def gauss(x, a, mu, sig):
    return a*np.exp(-(x-mu)**2/(2*sig**2))

param_ini = [200, 5900, 20]

# picname = '../data/fig/EnergySpectrum.png'
fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    squeeze=False,
    tight_layout=True,
    facecolor='whitesmoke',
    sharex=True,
    )

fig.supxlabel('Energy (eV)')
fig.supylabel('Counts')
# ax.grid(linestyle='--')
binnum = 70

for pixnum in range(4):
    counts, bins = np.histogram(energy[pixnum], bins=binnum, range=(5800,6000))
    bins = bins[:-1]
    popt, pcov = curve_fit(gauss, bins, counts, p0=param_ini)
    fitting = gauss(bins, popt[0], popt[1], popt[2])

    axes[pixnum//2, pixnum%2].errorbar(bins, counts, yerr=np.sqrt(counts), fmt=',k')
    axes[pixnum//2, pixnum%2].plot(bins, fitting, '-r')
    axes[pixnum//2, pixnum%2].set_title('pix.'+str(pixnum+1))

    print("Gaussian fit @5.9 keV for pix."+str(pixnum+1))
    print("param : a = %.2e, mu = %.2e eV, sigma = %.2e eV" % (popt[0], popt[1], popt[2]))
    print("")

# plt.savefig(picname,bbox_inches='tight',dpi=150)
plt.show()
