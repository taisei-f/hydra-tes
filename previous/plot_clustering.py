import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.ticker import ScalarFormatter
import matplotlib.cm as cm
import matplotlib.colors
import module.pixestimate as px
from sklearn.cluster import KMeans


picname = 'rise_vs_fall_pattern123.png'
featurefile = '../data/feature.hdf5'

# clustering
pxest = px.PixEstimate(featurefile)
rmvd_tau = pxest.rmvd_tau
rmvd_ph = pxest.rmvd_ph

############
# set data
############
data0 = rmvd_tau[:,0].reshape(-1,1)
data1 = rmvd_tau[:,1].reshape(-1,1)

data = np.hstack((data0, data1))
print(data)

# draw
rise_max = data[:,0].max()
rise_min = data[:,0].min()
ph_max = data[:,1].max()
ph_min = data[:,1].min()
fig, ax = plt.subplots(facecolor='whitesmoke')
# hist = ax.hist2d(data[:,0],data[:,1],bins=40,cmap=cm.jet)
ax.set_xlabel('20-80% rise time (s)')
ax.set_ylabel('80-20% fall time (s)')
# ax.set_xlim(rise_min-rise_max*0.1, rise_max*1.1)
# ax.set_ylim(ph_min-ph_max*0.1, ph_max*1.1)
# ax.set_xlim(0, 2.0*10**-4)
# ax.set_ylim(0, 1.0*10**-3)
ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci',axis='x',scilimits=(-4,-4))
ax.ticklabel_format(style='sci',axis='y',scilimits=(-3,-3))

ax.scatter(data[:,0], data[:,1],c=pxest.kmeans.labels_)
# fig.colorbar(hist[3],ax=ax)
plt.savefig(picname,bbox_inches='tight',dpi=150)
# plt.show()
