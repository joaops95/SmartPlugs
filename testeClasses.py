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

"""
We have 3 types of waves, 0 , 1, 2
"""
class WaveGenerator:
    def __init__(self, frequency, maxtime, df_train, df_test):
        self.frequency = frequency
        self.maxtime = maxtime
        self.numSamples = (maxtime/(1/frequency))
        self.time = np.linspace(0, maxtime, 2 * frequency, endpoint=False)
        self.df_train = df_train
        self.df_test = df_test
        self.wavesnumbers = 500       
        self.imgindex = 0
        self.imgindex1 = 0
        self.testEx = False
        self.arr1 = [] 
        self.arr2 = [] 
        self.arr3 = []
        self.testImgsPercentile = 0.01

    def generateWaves(self, df_train, df_test ,t, f, fs):
        try:
            os.makedirs('./imgs/test')
            os.makedirs('./imgs/train')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        for i in range(0,self.wavesnumbers):
            testEx = False  
            if(i >= (self.wavesnumbers - self.testImgsPercentile*self.wavesnumbers)):
                path1 = './imgs/test/img{number}.png'.format(number=self.imgindex1)
                self.imgindex1 = self.imgindex1 + 1
                testEx = True
            else:
                path1 = './imgs/train/img{number}.png'.format(number=imgindex)
            pure = random.randrange(2,5)*np.sin(self.frequency * 2 * np.pi * self.time)
            noise = np.random.normal(0, random.uniform(0.5,2.5), np.array(pure).shape)
            wave = pure + noise
            self.arr1.append(wave)
            toSpectrogram(arr1[i],fs,path1)
            imgResizeGrayScale(path1)
            if(testEx):
                df_test.loc[self.imgindex1] = [0, np.asarray(normalizeData(path1))]
            else:    
                df_train.loc[imgindex] = [0, np.asarray(normalizeData(path1))]
                imgindex = imgindex + 1


        for i in range(0,self.wavesnumbers):
            testEx = False  
            if(i >= (self.wavesnumbers - self.testImgsPercentile*self.wavesnumbers)):
                path1 = './imgs/test/img{number}.png'.format(number=imgindex1)
                imgindex1 = imgindex1 + 1
                testEx = True
            else:
                path1 = './imgs/train/img{number}.png'.format(number=imgindex)
            pure = random.randrange(5,10)*np.cos(self.frequency * 6 * np.pi * self.time)
            noise = np.random.normal(0, random.uniform(0.1,0.5), np.array(pure).shape)
            wave = pure + noise
            arr2.append(wave)
            toSpectrogram(arr2[i],fs,path1)
            imgResizeGrayScale(path1)
            if(testEx):
                df_test.loc[imgindex1] = [1, np.asarray(normalizeData(path1))]
            else:    
                df_train.loc[imgindex] = [1, np.asarray(normalizeData(path1))]
                imgindex = imgindex + 1

        
    for i in range(0,wavesnumbers):
        testEx = False  
        if(i >= (self.wavesnumbers - self.testImgsPercentile*self.wavesnumbers)):
            path1 = './imgs/test/img{number}.png'.format(number=imgindex1)
            imgindex1 = imgindex1 + 1
            testEx = True
        else:
            path1 = './imgs/train/img{number}.png'.format(number=imgindex)
        pure = random.randrange(2,5)*np.sin(f * 2 * np.pi * t)
        pure2 = random.randrange(5,10)*np.cos(f * random.randrange(2,6) * np.pi * t)
        noise = np.random.normal(0, random.uniform(0.1,0.5), np.array(pure).shape)
        wave = pure + pure2 + noise
        arr3.append(wave)
        toSpectrogram(arr3[i],fs,path1)
        imgResizeGrayScale(path1)
        if(testEx):
            df_test.loc[imgindex1] = [2, np.asarray(normalizeData(path1))]
        else:    
            df_train.loc[imgindex] = [2, np.asarray(normalizeData(path1))]
            imgindex = imgindex + 1
    df_train = shuffle(df_train)
    df_train.to_pickle("./dataframe_train.pkl")
    df_test.to_pickle("./dataframe_test.pkl")
    return arr1, arr2, arr3

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
    resized_image = cv2.resize(gray_image, (1, 250))
    cv2.imwrite(path,resized_image)
    print('img {path} resized'.format(path = path))
    
def normalizeData(self, path):
    normalizedArr = []
    unnormalized_arr = cv2.imread(path)
    for element in unnormalized_arr:
        normalized_value = np.mean(element) / 255.0
        #round(np.mean(element)/(255),3)
        normalizedArr.append(normalized_value)
    return normalizedArr
     

waves = WaveGenerator()