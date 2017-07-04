import sys
from subprocess import Popen, PIPE
from threading  import Thread
from Queue      import Queue, Empty
from subprocess import call
import binascii
import time
import signal
import matplotlib.mlab as mlab
import numpy as np
import pandas as pd
from scipy import signal
import json
from requests import *
import datetime
import pygame as pg # module that allows us to play music and change the volume regarding to alpha level
import math
from pyaudio import PyAudio
from functions import *
print sys.argv[1] + " range chosen" # you can add the argument when you call the
FreqRange = sys.argv[1]
if FreqRange == 'alpha':
    freqRange = np.array([8, 12])
elif FreqRange == 'gamma':
    freqRange = np.array([25, 50])
elif FreqRange == 'beta':
    freqRange = np.array([12, 25])
elif FreqRange == 'theta':
    freqRange = np.array([4, 7])
# script python communication_Demo.py alpha , or ask the user which freqRange to choose
# freqRange = input("choose between alpha, beta, gamma, delta, theta ? dont forget the \"\" ")

cpt = 0
buffersize = 200 # there are 200 points for the four channels, so 1000 at all for one second (dont forget the index number)
buffer_1 = []
nb_channels = 4
ind_2_remove_in_buffer1 = []
ind_channel_1 = []
ind_channel_2 = []
ind_channel_3 = []
ind_channel_4 = []
NFFT = 200
fs_hz = 200
overlap = 0
process = Popen(['/usr/local/bin/node', 'openBCIDataStream.js'], stdout=PIPE)
queue = Queue()
thread = Thread(target=enqueue_output, args=(process.stdout, queue))
thread.daemon = True # kill all on exit
thread.start()


# the following loop saves the index of the buffer that are interesting, without the channel id every 0 [5]
for ind in range(0, buffersize):
    # starts at index 0 which is the number of the sample
    ind_channel_1.append(ind*5+1)
    ind_channel_2.append(ind*5+2)
    ind_channel_3.append(ind*5+3)
    ind_channel_4.append(ind*5+4)

cpt2 = 0
newMean_uv = 0
data = np.zeros((nb_channels, buffersize))
sample_number = 0
oldMean_uv = 5E-13
mean_array_uv = np.array([])

while True:
    try:
        # the first while loop builds the buffer_1 for 1 second, data are the processed by the second loop
        while (cpt < buffersize*5)  :
            buffer_1.append(queue.get_nowait())
            ''' len(buffer_1) gives 500'''
            cpt += 1
            cpt2 = 0
        while cpt2 <1 :
            cpt2 += 1
            buffer_1_array = np.asarray(buffer_1, dtype=np.float64)

            ''' len(data_channel_1) gives 200 '''
            data[0, :] = buffer_1_array[ind_channel_1]
            data[1, :] = buffer_1_array[ind_channel_2]
            data[2, :] = buffer_1_array[ind_channel_3]
            data[3, :] = buffer_1_array[ind_channel_4]

            result = np.zeros(nb_channels)

            for channel in range(4):
                result[channel] = extract_freqbandmean(200, fs_hz, data[channel,:], freqRange[0], freqRange[1])

            print result
            print freqRange + " mean for each channel: \n CHANNEL 1:  ", result[0]
            cpt = 0
            oldMean_uv = newMean_uv
            buffer_1 = []

    except Empty:
        continue # do stuff
    else:

        str(buffer_1)
        #sys.stdout.write(char)
