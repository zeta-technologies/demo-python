import sys
from subprocess import Popen, PIPE
from threading  import Thread
from Queue      import Queue, Empty
from subprocess import call
import binascii
import time
import signal
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
from functions import *

FreqRange = 'alpha'
if FreqRange == '':
    logging.warning('No frequency passed as argument')

if FreqRange == 'alpha':
    freqRange = np.array([6, 11])
elif FreqRange == 'gamma':
    freqRange = np.array([25, 50])
elif FreqRange == 'beta':
    freqRange = np.array([12, 25])
elif FreqRange == 'theta':
    freqRange = np.array([4, 7])
elif FreqRange == 'XXII_beta':
    freqRange = np.array([15, 23])
elif FreqRange == 'XXII_gamma':
    freqRange = np.array([38, 40])

'''constants for streaming loop'''
cpt = 0
cpt2 = 0
buffersize = 200 # a bit more than one second of data,
buffer_1 = []
nb_channels = 4
ind_2_remove_in_buffer1 = []
ind_channel_1 = []
ind_channel_2 = []
ind_channel_3 = []
ind_channel_4 = []
OPB1_mean_array_uv = np.array([])
OPB1_data = np.zeros((nb_channels, buffersize))

''' Save buffer, to keep data records somewhere'''
saved_buffer = []

'''launch node process'''
process = Popen(['/usr/local/bin/node', 'openBCIDataStream.js'], stdout=PIPE)
queue = Queue()
thread = Thread(target=enqueue_output, args=(process.stdout, queue))
thread.daemon = True # kill all on exit
thread.start()


'''for the fft '''
length = 200
NFFT = 200
fs_hz = 200
# overlap = NFFT/2 # useless for now

'''Neurofeedback loop'''
# newMean = 0 # useless now
# oldMean = 5E-13 # useless now
mean_array_alpha = []
mean_array_delta = []
ratio_array = []

'''reorder channels index'''
# the following loop saves the index of the buffer that are interesting, without the channel id every 0 [nb_channels]
for ind in range(0, buffersize):
    # starts at index 0 which is the number of the sample
    ind_channel_1.append(ind*4)
    ind_channel_2.append(ind*4+1)
    ind_channel_3.append(ind*4+2)
    ind_channel_4.append(ind*4+3)

'''MAIN LOOP'''
while True:
    try:
        while (cpt < buffersize * nb_channels)  :
            buffer_1.append(queue.get_nowait())
            cpt += 1
            cpt2 = 0

        while cpt2 <1 :

            cpt2 += 1
            buffer_1_array = np.asarray(buffer_1)

            OPB1_data[0, :] = buffer_1_array[ind_channel_1]
            OPB1_data[1, :] = buffer_1_array[ind_channel_2]
            OPB1_data[2, :] = buffer_1_array[ind_channel_3]
            OPB1_data[3, :] = buffer_1_array[ind_channel_4]

            f_ch1 = filter_data(OPB1_data[0, :], fs_hz)
            f_ch2 = filter_data(OPB1_data[1, :], fs_hz)
            f_ch3 = filter_data(OPB1_data[2, :], fs_hz)
            f_ch4 = filter_data(OPB1_data[3, :], fs_hz)

            OPB1_bandmean_delta = np.zeros(nb_channels)
            OPB1_bandmean_alpha = np.zeros(nb_channels)

            OPB1_bandmax = np.zeros(nb_channels)
            OPB1_bandmin = np.zeros(nb_channels)

            for channel in range(4):
                OPB1_bandmean_alpha[channel] = extract_freqbandmean(200, fs_hz, OPB1_data[channel,:], 6, 11)
                OPB1_bandmean_delta[channel] = extract_freqbandmean(200, fs_hz, OPB1_data[channel,:], 1, 4)

            ''' Get the mean, min and max of the last result of all channels'''
            newMean_alpha = np.average(OPB1_bandmean_alpha)
            newMean_delta = np.average(OPB1_bandmean_delta)
            ratio = newMean_alpha / newMean_delta
            print 'ratio', ratio
            ''' increment the mean, min and max arrays of the freqRange studied'''
            ratio_array.append(ratio)

            ''' Freq is MAX = 1 500 when Ratio is Max, and freq is MIN = 500 when freqRange is MAX'''
            frequency = neurofeedback_freq(ratio_array, freqMax= 1500, freqMin= 500)
            # frequencyBis = nfFreqBis(ratio_array, freqMax = 1500, freqMin = 500)

            if np.invert(math.isnan(frequency)): #the issue is that the first frequencies are not defined, thus are NaN float. sine_tone works only with float
                print frequency
                sine_tone(frequency, 1, 160000)

            # pg.mixer.music.stop()
            cpt = 0
            buffer_1 = []
            saved_buffer.append([buffer_1])

    except Empty:
        continue # do stuff
    else:
        str(buffer_1)
        #sys.stdout.write(char)
