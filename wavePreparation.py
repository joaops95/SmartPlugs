import os, errno
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from scipy import fftpack
import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf
from keras.models import load_model
import datetime
import re


class WavePrepare:
    def __init__(self, path, wave, t):
        self.path = path
        self.wave = wave
        self.t = t
        self.fs = len(t)
        self.toSpectrogram(self.wave, self.fs, self.preparePath(self.path))
        self.imgResizeGrayScale(self.preparePath(self.path))
    
    def preparePath(self, path):
        if(len(os.listdir(path)) == 0):
            newpath = path + '/img{number}.png'.format(number=0)
            print(path)
            return newpath
        else:
            newest_img = os.listdir(path)[0]
            img_number = int(re.search(r'\d+', newest_img).group())
            newpath = path + '/img{number}.png'.format(number=img_number+1)
            return newpath
            
    # def toFFT(self, wave, fs, f):
    #     X = fftpack.fft(wave)
    #     freqs = fftpack.fftfreq(len(wave)) * fs
    #     return X, freqs

    def toSpectrogram(self, wave,fs,path):
        fig, ax = plt.subplots()
        ax.specgram(wave, Fs=fs)
        plt.axis('off')
        ax.set_position([0, 0, 1, 1])
        plt.savefig(path)
        plt.close()
        
    def imgResizeGrayScale(self, path):
        img = cv2.imread(path)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #resized_image = cv2.resize(gray_image, (1, 250))
        cv2.imwrite(path,gray_image)
        print('img {path} resized'.format(path = path))
        
    # def normalizeData(self, path):
    #     normalizedArr = []
    #     unnormalized_arr = cv2.imread(path)
    #     for element in unnormalized_arr:
    #         normalized_value = np.mean(element) / 255.0
    #         #round(np.mean(element)/(255),3)
    #         normalizedArr.append(normalized_value)
    #     return normalizedArr
