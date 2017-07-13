import pygame as pg
from pygame.locals import *
from constantesDataStream import *
import sys
from subprocess import Popen, PIPE
from threading  import Thread
from Queue import Queue, Empty
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
import math
import time
from pyaudio import PyAudio
from functions import *

'''GAME INIT'''
pg.init()
level = 0 # level initialized
score = 0
# x = 0
'''Load images, sonds libraries'''
levels_images = ['images/level0.png','images/level1.png','images/level2.png','images/level3.png','images/level4.png','images/level5.png','images/level6.png']
winImg = "images/win.png"
scoreDigitImages = ['images/0.png', 'images/1.png', 'images/2.png', 'images/3.png', 'images/4.png', 'images/5.png', 'images/6.png', 'images/7.png', 'images/8.png', 'images/9.png' ]
punchinball_images = ['images/punch1.png','images/punch2.png','images/punch3.png','images/punch4.png','images/punch5.png', ]
punch_noise = pg.mixer.Sound("songs/punch.ogg")

'''background'''
screen = pg.display.set_mode((1024, 576), RESIZABLE)
fond = pg.image.load('images/ring.jpg').convert()

'''Punching ball'''
punchBall = pg.image.load("images/punch3.png")
punchBall = pg.transform.scale(punchBall, (250, 450))

'''Score Bar'''
scoreBar = pg.image.load(levels_images[level]).convert_alpha()
scoreBar = pg.transform.scale(scoreBar, (90, 400))
scoreBar = pg.transform.rotate(scoreBar, -90)
# test = pg.image.load(levels_images[level]).convert_alpha()
# test =pg.transform.scale(scoreBar, (90, 400))

'''Winner image'''
winImg = pg.image.load(winImg).convert_alpha()
winImg = pg.transform.scale(winImg, (700, 440))
# punchBall = punch.set_colorkey((255,255,255))

'''Score digit '''
scoreTxt = pg.image.load('images/scoretxt.png')
scoreTxt = pg.transform.scale(scoreTxt, (150, 50))
scoreDigit = pg.image.load(scoreDigitImages[0])
scoreDigit = pg.transform.scale(scoreDigit, (70, 90))

'''Position everything on the screen'''
screen.blit(scoreTxt, (670,30))
screen.blit(fond, (0, 0))
screen.blit(punchBall, (350, -5))
screen.blit(scoreBar, (317, 460))
screen.blit(scoreDigit, (800, 30))
# screen.blit(test, (317, 460))
pg.display.flip()

'''launch node process'''
process = Popen(['/usr/local/bin/node', 'openBCIDataStream.js'], stdout=PIPE)
queue = Queue()
thread = Thread(target=enqueue_output, args=(process.stdout, queue))
thread.daemon = True # kill all on exit
thread.start()

'''MAIN LOOP'''

while True:
    for event in pg.event.get():
        if event.type == QUIT:
            pg.quit()
            sys.exit()
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

            OPB1_fdata[0, :] = filter_data(OPB1_data[0, :], fs_hz)
            OPB1_fdata[1, :] = filter_data(OPB1_data[1, :], fs_hz)
            OPB1_fdata[2, :] = filter_data(OPB1_data[2, :], fs_hz)
            OPB1_fdata[3, :] = filter_data(OPB1_data[3, :], fs_hz)

            # OPB1_bandmean_delta = np.zeros(nb_channels)
            OPB1_bandmean_alpha = np.zeros(nb_channels)

            OPB1_bandmax_alpha = np.zeros(nb_channels)
            OPB1_bandmin_alpha = np.zeros(nb_channels)

            for channel in range(4):
                OPB1_bandmean_alpha[channel] = extract_freqbandmean(200, fs_hz, OPB1_fdata[channel,:], 6, 11)
                # OPB1_bandmean_delta[channel] = extract_freqbandmean(200, fs_hz, OPB1_data[channel,:], 1, 4)

            ''' Get the mean, min and max of the last result of all channels'''
            newMean_alpha = np.average(OPB1_bandmean_alpha) #mean of the 4 channels, not the best metric I guess
            # newMean_delta = np.average(OPB1_bandmean_delta)
            # ratio = newMean_alpha / newMean_delta
            # print 'ratio', ratio
            ''' increment the mean, min and max arrays of the freqRange studied'''
            OPB1_mean_array_uv.append(newMean_alpha)

            if len(OPB1_mean_array_uv) != 0:
                delta = np.amax(OPB1_mean_array_uv) - np.min(OPB1_mean_array_uv) # Calculate delta before or after adding newMean_alpha?
            if len(OPB1_mean_array_uv) == 0:
                delta = 0
            print "new Mean", newMean_alpha
            print "range ", delta
            if delta == 0:
                level = 0

            if delta !=0:
                level = int(math.floor(7*(newMean_alpha-np.min(OPB1_mean_array_uv))/delta)) #we dont take the newMean

            if level == 7:

                score = score + 1
                punch_noise.play()
                # time.sleep(2)
                # angle = 5
                scoreDigit = pg.image.load(scoreDigitImages[score]).convert()
                scoreDigit = pg.transform.scale(scoreDigit, (70, 90))
                # movePunchinBall(angle, screen, scoreBar, scoreDigit, fond, punchBall)
                screen.blit(fond, (0, 0))
                screen.blit(scoreDigit, (800, 30))
                # screen.blit(punchBall, (350,-5))
                screen.blit(winImg, (100, 100))

            if level != 7:
                # angle = 5
                # movePunchinBall(angle, screen, scoreBar, scoreDigit, fond, punchBall)
                scoreBar = pg.image.load(levels_images[level]).convert_alpha()
                scoreBar = pg.transform.scale(scoreBar, (90, 400))
                scoreBar = pg.transform.rotate(scoreBar, -90)
                screen.blit(fond, (0, 0))
                screen.blit(punchBall, (350,-5))
                screen.blit(scoreBar, (317, 460))
                screen.blit(scoreDigit, (800, 30))
            print "level", level


                # Todo save all the mean to compare the last one to all the previous ones, if it's in the 80% of the highest it's a punch

            # if newMean_alpha > np.average(OPB1_bandmean_alpha[:-1]):
            #
            #     if level != 8:
            #         # x = x + 2
            #
            #         level_img = pg.image.load(levels_images[level-1]).convert_alpha()
            #         scaled_level = pg.transform.scale(level_img, (100, 440))
            #
            #         screen.blit(fond, (0, 0))
            #         screen.blit(punch, (200+2*x,-9))
            #         screen.blit(scaled_level, (700, 100))
            #         time.sleep(1)
            #         level = level + 1
            #
            #     else:
            #         level_img = pg.image.load(levels_images[level-1]).convert_alpha()
            #         # scaled_level = pg.transform.scale(level_img, (100, 440))
            #         punch_noise.play()
            #         screen.blit(fond, (0, 0))
            #         # screen.blit(punch, (200+2*x,-9))
            #         screen.blit(level_img, (100, 100))
            #         time.sleep(1)

                # punch(level, levels_images, fond, punch)

            pg.display.update()


            # pg.mixer.music.stop()
            # OPB1_mean_array_uv.append(newMean_alpha) # when it's append after, the level may be < 0...
            cpt = 0
            buffer_1 = []
            # saved_buffer.append([buffer_1])

    except Empty:
        continue # do stuff
    else:
        str(buffer_1)
        #sys.stdout.write(char)
