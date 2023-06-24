import numpy as np
import h5py
import matplotlib.pyplot as plt

file = '../data/feature.hdf5'
arb = 10**10

pha = []
temp = []
with h5py.File(file, 'r') as h5:
    # pha.append(h5['PHA']['pix0'][()])
    # pha.append(h5['PHA']['pix1'][()])
    # pha.append(h5['PHA']['pix2'][()])
    # pha.append(h5['PHA']['pix3'][()])

    temp.append(h5['clustering2']['template']['pix0'][()])
    temp.append(h5['clustering2']['template']['pix1'][()])
    temp.append(h5['clustering2']['template']['pix2'][()])
    temp.append(h5['clustering2']['template']['pix3'][()])

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    squeeze=False,
    tight_layout=True,
    facecolor='whitesmoke'
    )

# axes[1,1].hist(pha[0]/arb, bins=100)
# axes[1,1].set_title('pix.4')

# axes[0,0].hist(pha[1]/arb, bins=100)
# axes[0,0].set_title('pix.1')

# axes[0,1].hist(pha[2]/arb, bins=100)
# axes[0,1].set_title('pix.2')

# axes[1,0].hist(pha[3]/arb, bins=100)
# axes[1,0].set_title('pix.3')

axes[0,1].plot(temp[0])
axes[0,1].set_title('pix2')

axes[1,0].plot(temp[1])
axes[1,0].set_title('pix3')

axes[0,0].plot(temp[2])
axes[0,0].set_title('pix1')

axes[1,1].plot(temp[3])
axes[1,1].set_title('pix4')

fig.supxlabel('time')
# fig.supylabel('counts')

plt.show()
plt.savefig("../data/fig/template2.png",bbox_inches="tight")