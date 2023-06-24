import numpy as np
import h5py
import matplotlib.pyplot as plt

file = '../../data/feature.hdf5'
arb = 10**8

data = []
with h5py.File(file, 'r') as h5:
    data.append(h5["Saiteki"]["Pix0"]["PHA"][()])
    data.append(h5["Saiteki"]["Pix1"]["PHA"][()])
    data.append(h5["Saiteki"]["Pix2"]["PHA"][()])
    data.append(h5["Saiteki"]["Pix3"]["PHA"][()])

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    squeeze=False,
    tight_layout=True,
    facecolor='whitesmoke'
    )

for pixnum in range(4):
    # axes[pixnum//2, pixnum%2].plot(data[pixnum])
    axes[pixnum//2, pixnum%2].hist(data[pixnum]/arb, bins=100)
    axes[pixnum//2, pixnum%2].set_title('Pix'+str(pixnum))

fig.supxlabel('PHA')
fig.supylabel('counts')

plt.show()
# plt.savefig("../data/fig/template2.png",bbox_inches="tight")