import sys
from subprocess import Popen, PIPE
from threading  import Thread
from Queue      import Queue, Empty
from subprocess import call
import binascii
# import csv
# from scipy.cluster.vq import kmeans2, whiten
# from mpl_toolkits.mplot3d import Axes3D
# from numpy import genfromtxt
# import argparse
# from scipy.stats.stats import pearsonr
import time
import signal
#import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import pandas as pd
import scipy as sp
import heapq
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy import signal
import json
from requests import *
import datetime
import pygame as pg # module that allows us to play music and change the volume regarding to alpha level
import math
from pyaudio import PyAudio


#%matplotlib inline
cpt = 0
buffersize = 200
buffer_1 = []
ind_2_remove_in_buffer1 = []
ind_channel_1 = []
ind_channel_2 = []
ind_channel_3 = []
ind_channel_4 = []
length = 200
NFFT = 200
fs_hz = 200
overlap = NFFT - 30
#pg.mixer.init()
#pg.mixer.music.load('sound.mp3')

def filter_data(data, fs_hz):

    # filter the data to remove DC
    hp_cutoff_hz = 1.0
    b, a = signal.butter(2, hp_cutoff_hz / (fs_hz / 2.0), 'highpass')  # define the filter
    ff_data = signal.lfilter(b, a, data, 0)  # apply along the zeroeth dimension

    # filter from 5 to 35 Hz, helps remove 50Hz noise and replicates paper
    ## also helps remove the DC line noise (baseline drift)
    ## 125 is half the sampling rate (250Hz/2)
    b, a = signal.butter(4, (2.0 / (fs_hz / 2.0), 40.0 / (fs_hz / 2.0)), btype='bandpass')
    f_data = signal.lfilter(b, a, data, axis=0)

    # notch filter the data to remove 50 Hz and 100 Hz
    notch_freq_hz = np.array([50.0])  # these are the center frequencies
    for freq_hz in np.nditer(notch_freq_hz):  # loop over each center freq
        bp_stop_hz = freq_hz + 3.0 * np.array([-1, 1])  # set the stop band
        b, a = signal.butter(3, bp_stop_hz / (fs_hz / 2.0), 'bandstop')  # create the filter
        ff_data = signal.lfilter(b, a, f_data, 0)  # apply along the zeroeth dimension

    return f_data

def wave_amplitude(data, fs_hz, NFFT, overlap, length, wave_range):
    data = np.asarray(data)
    data = data[:length, :]
    f_eeg_data = filter_data(data, fs_hz)
    #t0 = time.time()
    if wave_range == 'alpha':
        wave_band_Hz = np.array([8, 12])  # where to look for the alpha peak

    elif wave_range == 'gamma':
        wave_band_Hz = np.array([25, 50])

    elif wave_range == 'beta':
        wave_band_Hz = np.array([12, 25])

    elif wave_range == 'theta':
        wave_band_Hz = np.array([4, 7])


    size = f_eeg_data.shape[1]  # verify that channels are in 1 position

    mean_range = np.zeros((1, size))


    max_range = np.zeros((1, size))
    min_range = np.zeros((1, size))

    ratio = np.zeros((1, size))

    for channel in range(size):
        # data[channel] = rolling_mean(data[channel])
        # data[channel] = data[channel] - np.mean(data[channel])
        # print(f_eeg_data.shape)

        # parse the data
        spec_PSDperHz, freqs, t_spec = mlab.specgram(f_eeg_data[25:, channel],
                                                     NFFT=NFFT,
                                                     Fs=fs_hz,
                                                     window=mlab.window_hanning,
                                                     noverlap=overlap
                                                     )

        # convert the units of the spectral data
        spec_PSDperBin = spec_PSDperHz * fs_hz / float(NFFT)  # convert to "Power Spectral Density per bin"
        spec_PSDperBin = np.asarray(spec_PSDperBin)
        # print(spec_PSDperBin.shape) # from 1 to 110 Hz, step of 1Hz

        # take the average spectrum according to the time - axis 1

        bool_inds_wave_range = (freqs > wave_band_Hz[0]) & (freqs < wave_band_Hz[1])

        spec_PSDperBin_range = spec_PSDperBin[bool_inds_wave_range]

        mean_range[0][channel] = np.mean(spec_PSDperBin_range)

        max_range[0][channel] = np.amax(spec_PSDperBin_range)

        # get the frequency of the max in each range alpha, beta, theta, gamma
        #max_range_idx = np.argmax(spec_PSDperBin_range)

        #freq_range = freqs[bool_inds_wave_range == 1]

        #freq_range = freq_range[max_range_idx]
        # print(freq_alpha, freq_beta, freq_theta, freq_gamma)

        # get the median of the 4 channels

    # print(max_beta)
    med_range = np.median(mean_range[0][:])

    max_range = np.amax(mean_range[0][:])

    min_range = np.min(mean_range[0][:])

    # ratio = med_beta / med_theta
    #time_last_event = time.time()-t0
    # print(mean_alpha)

    # return [med_alpha, max_alpha, min_alpha, freq_alpha,
    #        med_beta, max_beta, min_beta, freq_beta,
    #        med_theta, max_theta, min_theta, freq_theta,
    #        med_gamma, max_gamma, min_gamma, freq_gamma, time_last_alpha]

    # result = [med_range, max_range, min_range]
    result = med_range
    # print(med_gamma.type())

    return result

def enqueue_output(out, queue):
    while True:
        lines = out.readline()
        out.flush()
        queue.put(lines)

def sine_tone(freq, duration, bitrate):
    #See http://en.wikipedia.org/wiki/Bit_rate#Audio
    BITRATE = bitrate #number of frames per second/frameset.

    #See http://www.phy.mtu.edu/~suits/notefreqs.html
    FREQUENCY = freq #Hz, waves per second, 261.63=C4-note.
    LENGTH = duration #seconds to play sound

    NUMBEROFFRAMES = int(BITRATE * LENGTH)
    RESTFRAMES = NUMBEROFFRAMES % BITRATE
    WAVEDATA = ''
    # print (type(FREQUENCY))

    for x in xrange(NUMBEROFFRAMES):
        WAVEDATA += chr(int(math.sin(x / ((BITRATE / FREQUENCY) / math.pi)) * 127 + 128))
    #fill remainder of frameset with silence
    for x in xrange(RESTFRAMES):
        WAVEDATA += chr(128)

    p = PyAudio()
    stream = p.open(
        format=p.get_format_from_width(1),
        channels=1,
        rate=BITRATE,
        output=True,
        )
    stream.write(WAVEDATA)
    stream.stop_stream()
    stream.close()
    p.terminate()

process = Popen(['/usr/local/bin/node', 'openBCIDataStream.js'], stdout=PIPE)
queue = Queue()
thread = Thread(target=enqueue_output, args=(process.stdout, queue))
thread.daemon = True # kill all on exit
thread.start()



#        if len(self.buffer_1) < self.buffersize:
#            self.buffer_1.append([ch1, ch2, ch3, ch4])
#            self.count += 1
#        else:
            #filter_data(self.buffer_1, f_hz)
#            self.alpha_mean = (self.buffer_1, self.alpha_threshold, self.fs_hz, self.NFFT, self.overlap)
#            print(self.alpha_mean)
#            self.buffer_1 = []

# def sendDataServer(data_channel_1, data_channel_2, data_channel_3, data_channel_4):
#     data = {
#             'Channel 1': data_channel_1
#             'Channel 2': data_channel_2
#             'Channel 3': data_channel_3
#             'Channel 4': data_channel_4
#     }
    # response = post('https://blink-detector.herokuapp.com/eegs.json',json=data)
    # return response
def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

for ind in range(0, buffersize):
    #ind_2_remove_in_buffer1.append(ind*5)
    # starts at index 0 which is the number of the sample
    ind_channel_1.append(ind*5+1)
    ind_channel_2.append(ind*5+2)
    ind_channel_3.append(ind*5+3)
    ind_channel_4.append(ind*5+4)

cpt2 = 0
newMean_uv = 0
data = []
sample_number = 0
oldMean_uv = 5E-13
# VOLUME = 0.5
mean_array_uv = np.array([])
while True:
    try:
        while (cpt < buffersize * 5)  :
            buffer_1.append([queue.get_nowait()])
            cpt += 1
        #if len(buffer_1) == buffersize*5 :
            #for ind2 in sorted(ind_2_remove_in_buffer1, reverse = True):
            #    del buffer_1[ind2]
            #print buffer_1(ind_channel_1)
            #wave_amplitude()
            cpt2 = 0

        while cpt2 <1 :

            # print buffer_1
            # print len(buffer_1)
            cpt2 += 1
            # print type(buffer_1)
            buffer_1_array = np.asarray(buffer_1, dtype=np.float64)
            #print type(buffer_1)
            data_channel_1 = buffer_1_array[ind_channel_1]
            data_channel_2 = buffer_1_array[ind_channel_2]
            data_channel_3 = buffer_1_array[ind_channel_3]
            data_channel_4 = buffer_1_array[ind_channel_4]
            # print len(data_channel_4), len(data_channel_3), len(data_channel_2), len(data_channel_1)
            # print data_channel_2.dtype
            result1 = wave_amplitude(data_channel_1, fs_hz, NFFT, overlap, length, 'alpha' )
            result2 = wave_amplitude(data_channel_2, fs_hz, NFFT, overlap, length, 'alpha' )
            result3 = wave_amplitude(data_channel_3, fs_hz, NFFT, overlap, length, 'alpha' )
            result4 = wave_amplitude(data_channel_4, fs_hz, NFFT, overlap, length, 'alpha' )
            #oldMean = (result1 + result2 + result3 + result4)/4

            newMean_uv = np.average(result1 + result2 + result3 + result4)
            # newMean_uv = np.average( result2 )
            mean_array_uv.append(newMean_uv)

            # VOLUME = (newMean_uv-min(oldMean_uv,newMean_uv))/(max(oldMean_uv, newMean_uv)-min(oldMean_uv, newMean_uv))

            spread_average = np.average(mean_array_uv[-5:-1]) # the spread_average takes the 5 last Means in the array mean_array_uv, and get the mean of them
            BIG_MEAN = np.average(mean_array_uv) # the BIG MEAN is the global mean
            frequency = 1000/math.pi*np.arctan(spread_average*1e9-BIG_MEAN)+1000 # 1000/Pi * arctan(x-A) + 1000, gives frequency between 500 and 1500

            # frequency = np.float64(frequency).item()
            # frequency = round(frequency, 8)

            if np.invert(math.isnan(frequency)): #the issue is that the first frequencies are not defined, thus are NaN float. sine_tone works only with float
                print frequency
                sine_tone(frequency, 1, 160000)

            # print type(frequency)
            # sine_tone(frequency, 0.9, 16000)
            # pg.mixer.music.set_volume(VOLUME)
            # pg.mixer.music.play(0)
            # time.sleep(0.9)
            # pg.mixer.music.stop()
            # print "\n CHANNEL 1:  ", result1, "    CHANNEL 2 : ", result2, "    CHANNEL 3 : ", result3, "    CHANNEL 4 : ", result4
            print "\n    CHANNEL 2 : ", result2
            cpt = 0
            oldMean_uv = newMean_uv
            buffer_1 = []

#    req = Request('https://blink-detector.herokuapp.com/eegs.json')
#    req.add_header('Content-Type', 'application/json')
#
#    response = urlopen(req, json.dumps(data))

    except Empty:
        continue # do stuff
    else:
        # wave_amplitude(data, fs_hz, NFFT, overlap, 'alpha')
        str(buffer_1)
        #sys.stdout.write(char)

#a = numpy.array([1, 2, 3, 4], dtype=numpy.float64)
#a.astype(numpy.int64)
