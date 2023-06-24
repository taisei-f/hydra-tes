import numpy as np
import h5py
from sklearn.cluster import KMeans
from . import readhdf5


class PixEstimate:
    def __init__(self, featurefile):
        self.featurefile = featurefile
        with h5py.File(self.featurefile, 'r') as file:
            tau_list = file['feature']['tau'][()]
            ph_list = file['feature']['pulseheight'][()]

        tau_arr = np.array(tau_list)
        self.ph_arr = np.array(ph_list)

        # not nan -> True
        not_nan = ~np.isnan(tau_arr).any(axis=1)
        # not outlier -> True
        not_outlier = (tau_arr[:,0] < 2*10**(-4)) & (tau_arr[:,1] < 2*10**(-3))
        # data in pix -> True
        pix1_mask = (tau_arr[:,0]<0.1*10**(-4)) & (tau_arr[:,1]<0.2*10**(-3))
        pix2_mask = (tau_arr[:,0]>0.1*10**(-4)) & (tau_arr[:,0]<0.25*10**(-4)) & (tau_arr[:,1]>0.3*10**(-3)) & (tau_arr[:,1]<0.4*10**(-3))
        pix3_mask = (tau_arr[:,0]>0.5*10**(-4)) & (tau_arr[:,0]<0.75*10**(-4)) & (tau_arr[:,1]>0.5*10**(-3)) & (tau_arr[:,1]<0.7*10**(-3))
        pix4_mask = (tau_arr[:,0]>1.2*10**(-4)) & (tau_arr[:,0]<1.5*10**(-4)) & (tau_arr[:,1]>0.65*10**(-3)) & (tau_arr[:,1]<0.8*10**(-3))

        pix_mask = pix1_mask | pix2_mask | pix3_mask | pix4_mask

        self.mask = not_nan & not_outlier & pix_mask

        self.normal_eventnum = np.arange(len(tau_list))[self.mask]
        self.rmvd_tau = tau_arr[self.mask, :]
        self.rmvd_ph = self.ph_arr[self.mask, :]
        self.kmeans = KMeans(n_clusters=4).fit(self.rmvd_tau)

    def pixmask(self, pixnum):
        return self.kmeans.labels_ == pixnum
    
