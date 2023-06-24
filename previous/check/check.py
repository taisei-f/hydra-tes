import tesana
import h5py
from module import readhdf5
import numpy as np
from module import pixestimate as pxes
import matplotlib.pyplot as plt

rawfile = '../data/run041pn.hdf5'
feature = '../data/feature.hdf5'

raw = readhdf5.ReadHDF5(rawfile)
raw.OpenHDF5()
t = raw.time
p = raw.pulse
n = raw.noise

px = pxes.PixEstimate(feature)
raw_p = p[px.mask]
raw_n = n[px.mask]

fig, axes = plt.subplots(
    nrows = 2,
    ncols = 2,
    squeeze = False,
    tight_layout=True,
    facecolor='whitesmoke',
)

temp = []
for pixnum in range(4):
    print('Calculating pix.%d' % pixnum)
    print('Cluster center is :')
    print(px.kmeans.cluster_centers_[pixnum])
    print('')


    pixmask = px.pixmask(pixnum=pixnum)
    pix_pulse = raw_p[pixmask,:]
    pix_noise = raw_n[pixmask,:]

    tes = tesana.TES(t, pix_pulse, pix_noise)
    tes.PHA()
    temp.append(tes.tmpl)



axes[0,0].plot(t,temp[0])
axes[0,0].set_title('pix')

axes[0,1].plot(t,temp[1])
axes[0,1].set_title('pix')

axes[1,0].plot(t,temp[2])
axes[1,0].set_title('pix')

axes[1,1].plot(t,temp[3])
axes[1,1].set_title('pix')

fig.supxlabel('time')
# fig.supylabel('counts')

plt.show()
plt.savefig("../data/fig/template_check.png",bbox_inches="tight")
