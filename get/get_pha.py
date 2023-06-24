import numpy as np
import matplotlib.pyplot as plt
import h5py
from module import readhdf5

rawfile = '../../data/run041pn.hdf5'
feature = '../../data/feature.hdf5'
sampling_rate = 2.5e-8 # sec

###############################
# obtain normal event
###############################
raw = readhdf5.ReadHDF5(rawfile)
raw.OpenHDF5()
t = raw.time
p = raw.pulse
n = raw.noise


###############################
# functions
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


###############################
# template & PHA
###############################
pixmask = []
with h5py.File(feature, "a") as h5:
    # obtain mask
    pixmask.append(h5["Normal"]["pix0"]["Mask"][()])
    pixmask.append(h5["Normal"]["pix1"]["Mask"][()])
    pixmask.append(h5["Normal"]["pix2"]["Mask"][()])
    pixmask.append(h5["Normal"]["pix3"]["Mask"][()])

    del h5["Saiteki"]
    h5.create_group('Saiteki')

    for pixnum in range(4):
        print('Calculating PHA of pix%d' % pixnum)
        h5.create_group("Saiteki/Pix"+str(pixnum))

        # obtain raw data of one pixel
        pix_pulse = p[pixmask[pixnum], :]
        pix_noise = n[pixmask[pixnum], :]

        Nsamp = len(pix_noise[0])
        print("samples per 1 event : "+str(Nsamp))
        print("")
        fq = np.fft.fftfreq(Nsamp, sampling_rate)

        # average pulse
        ave_pulse = np.mean(pix_pulse, axis=0)
        h5.create_dataset(name='Saiteki/Pix'+str(pixnum)+"/AveragePulse", data=ave_pulse)
        print('written in : ~/Saiteki/Pix'+str(pixnum)+"/AveragePulse")

        # S/N ratio
        sn = np.sqrt(pulse_pow(ave_pulse)/noise_pow(pix_noise))
        h5.create_dataset(name='Saiteki/Pix'+str(pixnum)+"/SN", data=sn)
        print('written in : ~/Saiteki/Pix'+str(pixnum)+"/SN")

        # template
        tem = np.fft.ifft(np.fft.fft(ave_pulse)/noise_pow(pix_noise)).real
        norm = (np.max(ave_pulse)-np.min(ave_pulse))/(np.sum(tem*ave_pulse)/len(ave_pulse))
        template = tem*norm
        print("pix"+str(pixnum)+" template is obtained!")
        h5.create_dataset(name='Saiteki/Pix'+str(pixnum)+"/Template", data=template)
        print('written in : ~/Saiteki/Pix'+str(pixnum)+"/Template")

        # obtain PHA for each event
        pha = []
        for oneevent in pix_pulse:
            pha.append(np.sum(oneevent * template))

        print("The number of obtained PHA : %d" % len(pha))
        h5.create_dataset(name='Saiteki/Pix'+str(pixnum)+"/PHA", data=pha)
        print('written in : ~/Saiteki/Pix'+str(pixnum)+"/PHA")
        print("")

raw.CloseHDF5()
