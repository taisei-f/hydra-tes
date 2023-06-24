import numpy as np
import matplotlib.pyplot as plt
import h5py
from module import readhdf5 as read
from module import pixestimate as pxes

rawfile = '../data/run041pn.hdf5'
feature = '../data/feature.hdf5'
sampling_rate = 2.5e-8 # sec

###############################
# obtain normal event
###############################
raw = read.ReadHDF5(rawfile)
raw.OpenHDF5()
t = raw.time
p = raw.pulse
n = raw.noise

###############################
# remove anomalous events 
###############################
px = pxes.PixEstimate(featurefile=feature)
raw_p = p[px.mask]
raw_n = n[px.mask]

###############################
# clustering & template
###############################
# average of power spectrum of noise
def noise_pow(pix_n):
    fourier = []
    for oneevent in pix_n:
        fourier.append(np.fft.fft(oneevent))

    fourier = np.array(fourier)
    one_pow = np.abs(fourier)**2
    return np.mean(one_pow, axis=0)

# power spectrum of average pulse
def pulse_pow(ave_p):
    fourier= np.fft.fft(ave_p)
    return np.abs(fourier)**2

template = []
with h5py.File(feature, 'a') as h5:
    # h5.create_group('clustering2')
    # h5['clustering2'].create_group('PHA')
    # h5['clustering2'].create_group('template')

    for pixnum in range(4):
        print('Calculating PHA of pix.%d' % pixnum)
        print('Cluster center is :')
        print(px.kmeans.cluster_centers_[pixnum])
        print('')

        # obtain raw data of one pixel
        pixmask = px.pixmask(pixnum)
        pix_pulse = raw_p[pixmask,:]
        pix_noise = raw_n[pixmask,:]
        pix_ph = px.rmvd_ph[pixmask,:]

        Nsamp = len(pix_noise[0])
        fq = np.fft.fftfreq(Nsamp, sampling_rate)

        ave_pulse = np.mean(pix_pulse, axis=0)

        # S/N ratio
        sn = np.sqrt(pulse_pow(ave_pulse)/noise_pow(pix_noise))

        # template
        tem = np.fft.ifft(np.fft.fft(ave_pulse)/noise_pow(pix_noise)).real
        norm = (np.max(ave_pulse)-np.min(ave_pulse))/(np.sum(tem*ave_pulse)/len(ave_pulse))
        template.append(tem*norm)

        # calculate PHA
        pha = []
        for oneevent in pix_pulse:
            pha.append(np.sum(oneevent * template))
        
        print('file writing ...')
        h5.create_dataset(name='feature/PHA/pix'+str(pixnum), data=pha)
        h5.create_dataset(name='feature/PHA/template/pix'+str(pixnum), data=tem*norm)
        print('success')
        print('')

raw.CloseHDF5()
