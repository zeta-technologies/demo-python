# coding=utf-8

import pygame as pg
from pygame.locals import *
from constantes_PunchinBall import *
from constantesDataStream import *
import sys
from subprocess import Popen, PIPE
from threading  import Thread
# from Queue import Queue, Empty
# from subprocess import call
# import binascii
# import time
# import signal
# import numpy as np
# import pandas as pd
# import scipy as sp
# import heapq
# from scipy.interpolate import UnivariateSpline
# from scipy.interpolate import interp1d
# from scipy import signal
# import json
from requests import *
# import datetime
# import math
# import time
# from pyaudio import PyAudio

from functions import *
'''background'''
screen = pg.display.set_mode((1024, 576), RESIZABLE)
fond = pg.image.load(image_ring).convert()

'''Punching ball'''
punchBall = pg.image.load(punchBallImage)
punchBall = pg.transform.scale(punchBall, (250, 450))

'''Score Bar'''
scoreBar = pg.image.load(levels_images[level]).convert_alpha()
scoreBar = pg.transform.scale(scoreBar, (90, 400))
scoreBar = pg.transform.rotate(scoreBar, -90)
# test =pg.transform.scale(scoreBar, (90, 400))

'''Winner image'''
winImg = pg.image.load(winImg).convert_alpha()
winImg = pg.transform.scale(winImg, (700, 440))
# punchBall = punch.set_colorkey((255,255,255))

'''Score digit '''
scoreTxt = pg.image.load(image_score)
scoreTxt = pg.transform.scale(scoreTxt, (150, 50))
scoreDigit = pg.image.load(scoreDigitImages[0])
scoreDigit = pg.transform.scale(scoreDigit, (70, 90))

'''Fly game'''
sky = pg.image.load(skyImage).convert()
sky = pg.transform.scale(sky, (1024, 576))
cloud = pg.image.load(cloudImage).convert()
plane = pg.image.load(planeImage)
# plane = plane.set_colorkey((255, 255, 255))

'''launch node process'''
process = Popen(['/usr/local/bin/node', 'openBCIDataStream.js'], stdout=PIPE)
queue = Queue()
thread = Thread(target=enqueue_output, args=(process.stdout, queue))
thread.daemon = True # kill all on exit
thread.start()

'''REsting state'''
timerImage = pg.image.load(timer[0])
restingStateImage = pg.image.load(restingState).convert()

'''MAIN LOOP'''
continuer = 1
while continuer:

    home = pg.image.load(image_home).convert() #TODO add image_home
    home = pg.transform.scale(home, (1024, 576))
    screen.blit(home, (0,0))
    pg.display.flip()

    punchinBall = 1
    homeOn = 1
    fly = 1
    restingState = 1


    while homeOn:
        pg.time.Clock().tick(60)

        for event in pg.event.get():
            if event.type == QUIT:
                pg.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                if event.key == K_1:
                    homeOn = 0
                    punchinBall = 1
                    fly = 0
                    restingState = 0
                elif event.key == K_2:
                    homeOn = 0
                    punchinBall = 0
                    fly = 1
                    restingState = 0
                elif event.key == K_RETURN:
                    homeOn = 0
                    punchinBall = 0
                    fly = 0
                    restingState = 1

    if punchinBall :
        # Chargement du fond

        '''Position everything on the screen'''
        screen.blit(scoreTxt, (670, 30))
        screen.blit(fond, (0, 0))
        screen.blit(punchBall, (350, -5))
        screen.blit(scoreBar, (317, 460))
        screen.blit(scoreDigit, (800, 30))
        # screen.blit(test, (317, 460))
        pg.display.flip()
        # punch_noise = pg.mixer.Sound("songs/punch.ogg") # TODO resolve the MemoryError due to pg.mixer.Sound


        # Génération d'un niveau à partir d'un fichier
        # game = Game(gameChoice) # TODO create Game class
        # game.generer()
        # game.afficher(screen)
        #
        # # Création de  Kong
        # gauge = Level("images/dk_droite.png", "images/dk_gauche.png",
        #            "images/dk_haut.png", "images/dk_bas.png", niveau) # TODO create Level class

    if fly:
        '''Position everything on the screen'''
        screen.blit(sky, (0, 0))
        screen.blit(cloud, (800, 100))
        screen.blit(plane, (300, 200))
        # screen.blit(scoreBar, (317, 460))
        # screen.blit(scoreDigit, (800, 30))
        # screen.blit(test, (317, 460))
        pg.display.flip()

    if restingState:
        # timerImage = pg.image.load(timer[]).convert()
        screen.blit(restingStateImage, (0,0))
        screen.blit(timerImage, (0,0))
        pg.display.flip()
        # sec = sec + 1
        # print sec

    while punchinBall:

        pg.time.Clock().tick(30)
        for event in pg.event.get():
            if event.type == QUIT:
                pg.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    punchinBall = 0

        try:
            while cpt < buffersize * nb_channels :
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
                    delta = np.amax(OPB1_mean_array_uv) - np.min(OPB1_mean_array_uv)  # Calculate delta before or after adding newMean_alpha?
                if len(OPB1_mean_array_uv) == 0:
                    delta = 0
                print "new Mean of 4 channels", newMean_alpha
                print "Max - Min ", delta

                if delta == 0:
                    level = 0

                if delta !=0:
                    level = int(math.floor(7*(newMean_alpha-np.min(OPB1_mean_array_uv))/delta)) #we dont take the newMean

                if level == 7:
                    score = score + 1
                    # punch_noise.play()
                    scoreDigit = pg.image.load(scoreDigitImages[score]).convert()
                    scoreDigit = pg.transform.scale(scoreDigit, (70, 90))
                    screen.blit(fond, (0, 0))
                    screen.blit(scoreDigit, (800, 30))
                    screen.blit(winImg, (100, 100))

                if level != 7:
                    scoreBar = pg.image.load(levels_images[level]).convert_alpha()
                    scoreBar = pg.transform.scale(scoreBar, (90, 400))
                    scoreBar = pg.transform.rotate(scoreBar, -90)
                    screen.blit(fond, (0, 0))
                    screen.blit(punchBall, (350,-5))
                    screen.blit(scoreBar, (317, 460))
                    screen.blit(scoreDigit, (800, 30))
                print "level", level

                pg.display.update()

                cpt = 0
                buffer_1 = []

        except Empty:
            continue # do stuff
        else:
            str(buffer_1)
            #sys.stdout.write(char)

    while fly:

        pg.time.Clock().tick(60)
        for event in pg.event.get():
            if event.type == QUIT:
                pg.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    fly = 0

        try:
            while cpt < buffersize * nb_channels:
                buffer_1.append(queue.get_nowait())
                cpt += 1
                cpt2 = 0

            while cpt2 < 1:

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
                    OPB1_bandmean_alpha[channel] = extract_freqbandmean(200, fs_hz, OPB1_fdata[channel, :], 6, 11)
                    # OPB1_bandmean_delta[channel] = extract_freqbandmean(200, fs_hz, OPB1_data[channel,:], 1, 4)

                ''' Get the mean, min and max of the last result of all channels'''
                newMean_alpha = np.average(OPB1_bandmean_alpha)  # mean of the 4 channels, not the best metric I guess
                OPB1_mean_array_uv.append(newMean_alpha)
                maxAlpha = np.amax(OPB1_mean_array_uv)
                minAlpha = np.min(OPB1_mean_array_uv)

                if newMean_alpha == maxAlpha:
                    newPosy = 100
                elif newMean_alpha == minAlpha:
                    newPosy = 400
                else:
                    a = 1. * 300 / (minAlpha - maxAlpha)
                    b = 250 - 1. * (maxAlpha + minAlpha) * a / 2
                    newPosy = a * newMean_alpha + b

                # screen.blit(fond, (0, 0))
                deltaPosy = (newPosy - oldPosy) / steps
                screen.blit(sky, (0, 0))
                # screen.blit(plane, (300, oldPosy + deltaPosy * step))
                screen.blit(plane, (300, newPosy))
                print oldPosy, newPosy
                # pg.time.delay(400)
                pg.display.flip()

                print "new Mean of 4 channels", newMean_alpha, maxAlpha, minAlpha

                # scoreBar = pg.image.load(levels_images[level]).convert_alpha()
                # scoreBar = pg.transform.scale(scoreBar, (90, 400))
                # scoreBar = pg.transform.rotate(scoreBar, -90)
                oldPosy = newPosy
                cpt = 0
                buffer_1 = []

                # except Empty:
                #     continue  # do stuff
                # # else:
                #     # str(buffer_1)
                #     # sys.stdout.write(char)




        except Empty:
            continue # do stuff
        # else:
        #     str(buffer_1)
        #     #sys.stdout.write(char)

    while restingState:
        # pg.time.Clock().tick(30)

        for event in pg.event.get():
            if event.type == QUIT:
                pg.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    restingState = 0

        # time.sleep(1)
        pg.time.delay(990)
        # print sec
        if sec >= 100:
            timerSec = pg.image.load(timer[int(str(sec)[2])]).convert()
            timerDiz = pg.image.load(timer[int(str(sec)[1])]).convert()
            timerCen = pg.image.load(timer[int(str(sec)[0])]).convert()
            screen.blit(timerSec, (230, 0))
            screen.blit(timerDiz, (115, 0))
            screen.blit(timerCen, (0, 0))

        elif sec < 10:
            timerSec = pg.image.load(timer[int(str(sec)[0])]).convert()
            screen.blit(timerSec, (0, 0))

        elif sec >= 10 & sec < 100:
            timerSec = pg.image.load(timer[int(str(sec)[1])]).convert()
            timerDiz = pg.image.load(timer[int(str(sec)[0])]).convert()
            screen.blit(timerSec, (115, 0))
            screen.blit(timerDiz, (0, 0))


        # timerSec = pg.image.load(timer[sec%10]).convert()
        # timerDiz = pg.image.load(timer[int(math.floor(sec/10))%100]).convert()
        # timerCen = pg.image.load(timer[int(math.floor(sec/100))]).convert()
        # checkImp() # TODO write check impedances function

        pg.display.update()
        sec = sec + 1
