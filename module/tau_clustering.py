
#########################################
# THIS SCRIPT USES tau FOR CLUSTERING
#########################################

import numpy as np
import h5py
from sklearn.cluster import KMeans

feature = '../data/feature.hdf5'


########################################
# get data using for clustering
########################################
with h5py.File(feature, 'a') as h5:
    tau_list = h5['tau'][()]
    # ph_list = file['feature']['ph'][()]

    tau_arr = np.array(tau_list)
    # ph_arr = np.array(ph_list)


########################################
# data cleaning to get rmvd_tau
########################################
    alleventnum = len(tau_list)
    h5.create_group("Anomalous")
    h5.create_group("Normal")

    # nan -> True
    h5.create_group("Anomalous/Nan")
    nan = np.isnan(tau_arr).any(axis=1)
    nantau_eventnumbers = np.arange(alleventnum)[nan]
    print("cannot obtain tau :")
    print(nantau_eventnumbers)
    print(nantau_eventnumbers.shape)
    h5.create_dataset(name="Anomalous/TauIsNan/EventNumbers", data=nantau_eventnumbers)
    h5.create_dataset(name="Anomalous/TauIsNan/Mask", data=nan)
    print("written in : ~/Anomalous/TauIsNan")
    print("")

    # large outlier -> True
    h5.create_group("Anomalous/LargeOutlier")
    large_outlier = (tau_arr[:,0] > 2*10**(-4)) & (tau_arr[:,1] > 2*10**(-3))
    large_outlier_eventnumbers = np.arange(alleventnum)[large_outlier]
    print("large outlier :")
    print(large_outlier_eventnumbers)
    print(large_outlier_eventnumbers.shape)
    h5.create_dataset(name="Anomalous/LargeOutlier/EventNumbers", data=large_outlier_eventnumbers)
    h5.create_dataset(name="Anomalous/LargeOutlier/Mask", data=large_outlier)
    print("written in : ~/Anomalous/LargeOutlier")
    print("")

    # normal value -> True
    h5.create_group("Anomalous/SmallOutlier")
    pix0_mask = (tau_arr[:,0]<0.1*10**(-4)) & (tau_arr[:,1]<0.2*10**(-3))
    pix1_mask = (tau_arr[:,0]>0.1*10**(-4)) & (tau_arr[:,0]<0.25*10**(-4)) & (tau_arr[:,1]>0.3*10**(-3)) & (tau_arr[:,1]<0.4*10**(-3))
    pix2_mask = (tau_arr[:,0]>0.5*10**(-4)) & (tau_arr[:,0]<0.75*10**(-4)) & (tau_arr[:,1]>0.5*10**(-3)) & (tau_arr[:,1]<0.7*10**(-3))
    pix3_mask = (tau_arr[:,0]>1.2*10**(-4)) & (tau_arr[:,0]<1.5*10**(-4)) & (tau_arr[:,1]>0.65*10**(-3)) & (tau_arr[:,1]<0.8*10**(-3))
    pix_mask = pix0_mask | pix1_mask | pix2_mask | pix3_mask

    small_outlier = ~(nan | large_outlier | pix_mask)
    small_outlier_eventnumbers = np.arange(alleventnum)[small_outlier]
    print("small outlier :")
    print(small_outlier_eventnumbers)
    print(small_outlier_eventnumbers.shape)
    h5.create_dataset(name="Anomalous/SmallOutlier/EventNumbers", data=small_outlier_eventnumbers)
    h5.create_dataset(name="Anomalous/SmallOutlier/Mask", data=small_outlier)
    print("written in : ~/Anomalous/SmallOutlier")
    print("")


    h5.create_group("Normal/All")
    normal_eventnumbers = np.arange(alleventnum)[pix_mask]
    print("normal events :")
    print(normal_eventnumbers)
    print(normal_eventnumbers.shape)
    h5.create_dataset(name="Normal/All/EventNumbers", data=normal_eventnumbers)
    h5.create_dataset(name="Normal/All/Mask", data=pix_mask)
    print("written in : ~/Normal/All")
    print("")


    rmvd_tau = tau_arr[pix_mask, :]
    # rmvd_ph = ph_arr[mask, :]


########################################
# clustering
########################################
    kmeans = KMeans(n_clusters=4).fit(rmvd_tau)
    risetime_of_each_clst = np.array([kmeans.cluster_centers_[0,0],
                                      kmeans.cluster_centers_[1,0],
                                      kmeans.cluster_centers_[2,0],
                                      kmeans.cluster_centers_[3,0]])
    order = np.argsort(risetime_of_each_clst)

    anomalous = np.sort(
        np.hstack([
            nantau_eventnumbers,
            large_outlier_eventnumbers,
            small_outlier_eventnumbers]
        )
    )

    alllabel = kmeans.labels_
    for key in anomalous:
        alllabel = np.insert(alllabel, key, -1)

    h5.create_group("Normal/pix0")
    pix0_label = alllabel == order[0]   # mask
    pix0_eventnumbers = np.arange(alleventnum)[pix0_label]
    print("pix0 events :")
    print(pix0_eventnumbers.shape)
    h5.create_dataset(name="Normal/pix0/EventNumbers", data=pix0_eventnumbers)
    h5.create_dataset(name="Normal/pix0/Mask", data=pix0_label)
    print("written in : ~/Normal/pix0")
    print("")

    h5.create_group("Normal/pix1")
    pix1_label = alllabel == order[1]   # mask
    pix1_eventnumbers = np.arange(alleventnum)[pix1_label]
    print("pix1 events :")
    print(pix1_eventnumbers.shape)
    h5.create_dataset(name="Normal/pix1/EventNumbers", data=pix1_eventnumbers)
    h5.create_dataset(name="Normal/pix1/Mask", data=pix1_label)
    print("written in : ~/Normal/pix1")
    print("")

    h5.create_group("Normal/pix2")
    pix2_label = alllabel == order[2]   # mask
    pix2_eventnumbers = np.arange(alleventnum)[pix2_label]
    print("pix2 events :")
    print(pix2_eventnumbers.shape)
    h5.create_dataset(name="Normal/pix2/EventNumbers", data=pix2_eventnumbers)
    h5.create_dataset(name="Normal/pix2/Mask", data=pix2_label)
    print("written in : ~/Normal/pix2")
    print("")

    h5.create_group("Normal/pix3")
    pix3_label = alllabel == order[3]   # mask
    pix3_eventnumbers = np.arange(alleventnum)[pix3_label]
    print("pix3 events :")
    print(pix3_eventnumbers.shape)
    h5.create_dataset(name="Normal/pix3/EventNumbers", data=pix3_eventnumbers)
    h5.create_dataset(name="Normal/pix3/Mask", data=pix3_label)
    print("written in : ~/Normal/pix3")
    print("")
        
