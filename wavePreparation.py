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
import json
import librosa
import librosa.display


class WavePrepare:
    def __init__(self, wave):
        self.path = './imgs/train'
        self.wave = wave
        self.fs = len(wave)

    def preparePath(self):
        if(len(os.listdir(self.path)) == 0):
            newpath = self.path + '/img{number}.png'.format(number=0)
            print(self.path)
            return newpath
        else:
            newpath = self.path + '/img{number}.png'.format(number=len(os.listdir(self.path))+1)
            return newpath
            

    def toSpectrogram(self, wave,path):
        n_fft = len(wave)
        hop_length = int(len(wave)/len(wave))
        D = np.abs(librosa.stft(np.asarray(wave), n_fft=n_fft,  
                                hop_length=hop_length))
        _, ax = plt.subplots()
        DB = librosa.amplitude_to_db(D, ref=np.max)
        librosa.display.specshow(DB, sr=len(wave), hop_length=hop_length, 
                                x_axis='time', y_axis='log')
        plt.axis('off')
        ax.set_position([0, 0, 1, 1])
        plt.savefig(path)
        plt.close()
        
    def imgResizeGrayScale(self, path):
        img = cv2.imread(path)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        #resized_image = cv2.resize(gray_image, (50, 250))
        cv2.imwrite(path,gray_image)
        print('img {path} resized'.format(path = path))

    def addToDataSet(self, path, label, wave, x_efvalue ,train, train_path, test_path):
        if(train):
            df_train = pd.read_pickle(train_path)
            df_train.loc[len(df_train)] = [label, np.asarray(wave), x_efvalue]
            df_train.to_pickle(train_path)
            print(df_train)
        else:
            #, np.array(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY))/255
            df_test = pd.read_pickle(test_path)
            df_test.loc[len(df_train)] = [label, np.asarray(wave), x_efvalue]
            df_test.to_pickle(test_path)
    
    def createDataSets(self, path, ylabels, x_pure, x_efvalue):
        df_train = pd.DataFrame(columns=[ylabels, x_pure, x_efvalue])
        df_test = pd.DataFrame(columns=[ylabels, x_pure, x_efvalue])
        train_path = str(path) + "/dataframe_train.pkl"
        test_path = str(path) + "/dataframe_test.pkl"
        df_train.to_pickle(train_path)
        df_test.to_pickle(test_path)
        return train_path, test_path
    
    def createDummyData(self, qqty, wave, label, train_path):
        arr = []
        df_train = pd.read_pickle(train_path)
        for _ in range(0,qqty):
            noise = np.random.normal(0, random.uniform(0.1,0.5), np.array(wave).shape)
            wave_final = wave + noise
            arr.append(wave_final)
            path = self.preparePath()
            self.toSpectrogram(wave, path)
            self.imgResizeGrayScale(path)
            df_train.loc[len(df_train)] = [label, np.array(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY))/255, np.asarray(wave)]
            print(df_train)
        df_train.to_pickle(train_path)
            
# with open('./nodes.json') as json_file:
#     data = json.load(json_file)
#     for p in data:
#         nodeid = p['nodeid']
#         wave = p['lastwave']
# wave = [1, 2]
# waveprep = WavePrepare(wave)
# train_path, test_path = waveprep.createDataSets('.','label', 'x_pure', 'x_efvalue')
    train_path = './dataframe_train.pkl'
# #     test_path = '/dataframe_test.pkl'
# #     #waveprep.addToDataSet('./imgs/train/img21.png',1,wave, 0.5,True,train_path,test_path)
# #     #waveprep.createDummyData(20, wave, 2, train_path)
    df_train = pd.read_pickle(train_path)
    for i in range(0, len(df_train)):
        if(df_train['label'][i] == 2):
            plt.plot(df_train['x_pure'][i])
            plt.show()

        
    print(df_train)
    # plt.plot(df_train['x_pure'][0])
    # plt.show()
    # print(df_train)