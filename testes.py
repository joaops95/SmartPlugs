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
from tensorflow.keras import datasets, layers, models
import datetime
from keras.preprocessing import image


"""
We have 3 types of waves, 0 , 1, 2
"""


def generateWaves(df_train, df_test ,t, f, fs): # this function generates 3 waves
    testEx = False
    arr1 = [] 
    arr2 = [] 
    arr3 = []
    imgindex = 0 
    imgindex1 = 0
    wavesnumbers = 500
    testImgsPercentile = 0.2
    try:
        os.makedirs('./imgs/test')
        os.makedirs('./imgs/train')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    for i in range(0,wavesnumbers):
        testEx = False  
        if(i >= (wavesnumbers - testImgsPercentile*wavesnumbers)):
            path1 = './imgs/test/img{number}.png'.format(number=imgindex1)
            imgindex1 = imgindex1 + 1
            testEx = True
        else:
            path1 = './imgs/train/img{number}.png'.format(number=imgindex)
        pure = random.randrange(2,5)*np.sin(f * 2 * np.pi * t)
        noise = np.random.normal(0, random.uniform(0.5,2.5), np.array(pure).shape)
        wave = pure + noise
        arr1.append(wave)
        toSpectrogram(arr1[i],fs,path1)
        imgResizeGrayScale(path1)
        if(testEx):
            df_test.loc[imgindex1] = [0, np.array(cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2GRAY))/255]
        else:    
            df_train.loc[imgindex] = [0, np.array(cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2GRAY))/255]
            imgindex = imgindex + 1


    for i in range(0,wavesnumbers):
        testEx = False  
        if(i >= (wavesnumbers - testImgsPercentile*wavesnumbers)):
            path1 = './imgs/test/img{number}.png'.format(number=imgindex1)
            imgindex1 = imgindex1 + 1
            testEx = True
        else:
            path1 = './imgs/train/img{number}.png'.format(number=imgindex)
        pure = random.randrange(5,10)*np.cos(f * 6 * np.pi * t)
        noise = np.random.normal(0, random.uniform(0.1,0.5), np.array(pure).shape)
        wave = pure + noise
        arr2.append(wave)
        toSpectrogram(arr2[i],fs,path1)
        imgResizeGrayScale(path1)#print(np.asarray(arr1).shape)
        if(testEx):
            df_test.loc[imgindex1] = [1, np.array(cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2GRAY))/255]
        else:    
            df_train.loc[imgindex] = [1, np.array(cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2GRAY))/255]
            imgindex = imgindex + 1

        
    for i in range(0,wavesnumbers):
        testEx = False  
        if(i >= (wavesnumbers - testImgsPercentile*wavesnumbers)):
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
            df_test.loc[imgindex1] = [2, np.array(cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2GRAY))/255]
        else:    
            df_train.loc[imgindex] = [2, np.array(cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2GRAY))/255]
            imgindex = imgindex + 1
    df_train = shuffle(df_train)
    df_train.to_pickle("./dataframe_train.pkl")
    df_test.to_pickle("./dataframe_test.pkl")
    return arr1, arr2, arr3

def toFFT(wave, fs, f):
    X = fftpack.fft(wave)
    freqs = fftpack.fftfreq(len(wave)) * fs

    return X, freqs

def toSpectrogram(wave,fs,path):
    fig, ax = plt.subplots()
    ax.specgram(wave, Fs=fs)
    plt.axis('off')#print(np.asarray(arr1).shape)
    ax.set_position([0, 0, 1, 1])
    plt.savefig(path)
    plt.close()
    
def imgResizeGrayScale(path):
    img = cv2.imread(path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (50, 250))
    cv2.imwrite(path,resized_image)
    print('img {path} resized'.format(path = path))


def reshapeArr(df):
    temp = []
    for i in range(0, len(df)):
        temp.append(df.values[i][1])
    return np.asarray(temp, dtype=np.float64)

def reshapeArr1(df):
    temp = []
    for i in range(0, len(df)):
        temp.append(df.values[i][0])
    return np.asarray(temp, dtype=np.int64)

def createModel():
    model = models.Sequential()
    model.add(layers.Conv2D(128, (3, 3), activation='relu', batch_input_shape=(1200, 250, 50, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    return model

def testModel(x_test, pos):
    checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model = createModel()
    model.load_weights(latest)
    # Re-evaluate the model
    loss, acc = model.evaluate(x_test,  y_test, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))
    predictions = model.predict(x_test)
    print(np.argmax(predictions[pos]))
    #print(predictions[0])
    print(y_test[pos])
    
def trainModel(x_train, y_train):
    checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1,
                                                    period = 2)

    #Save the weights using the `checkpoint_path` format
    model = createModel()
    model.save_weights(checkpoint_path.format(epoch=0))

    history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test,y_test),
                    callbacks=[cp_callback])
    
    return history

def preProcessData(df_train, df_test):
    y_train = reshapeArr1(df_train)
    y_train = tf.reshape(y_train, (-1,1))
    x_train = reshapeArr(df_train)
    x_train = tf.reshape(x_train,(-1, 250, 50, 1))
    y_test = reshapeArr1(df_test)
    y_test = tf.reshape(y_test, (-1,1))
    x_test = reshapeArr(df_test)
    x_test = tf.reshape(x_test,(-1, 250, 50, 1))
    return y_train, x_train, y_test, x_test

def plotSetting(history, path):
    history_dict = history.history
    history_dict.keys()
    print(history_dict)
    acc = history_dict['accuracy']
    loss = history_dict['loss']

    epochs = range(1, len(acc) + 1)
    plt.figure
    # "-r^" is for solid red line with triangle markers.
    plt.plot(epochs, loss, '-r^', label='Training loss')
    # "-b0" is for solid blue line with circle markers.
    #plt.plot(epochs, val_loss, '-bo', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.figure
    plt.plot(epochs, acc, '-g^', label='Training acc')
    #plt.plot(epochs, val_acc, '-bo', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.savefig(path)
    plt.close()

def pfrint():
    print('ajajaj')
if __name__ == "__main__":
    f = random.uniform(49.5,50.5)  # Frequency, in cycles per second, or Hertz
    df_train = pd.DataFrame(columns=['y_train', 'x_train'])
    df_test = pd.DataFrame(columns=['y_test', 'x_test'])
    #f = 50
    fs = 50  # Sampling rate, or number of measurements per second
    maxtime = 0.08
    numSamples = (maxtime/(1/f))
    t = np.linspace(0, maxtime, 2 * fs, endpoint=False)

    arr1, arr2, arr3 = generateWaves(df_train,df_test ,t, f, f)
    
    
    #guarda valores df no em pkl
    df_train = pd.read_pickle("./dataframe_train.pkl")
    df_test = pd.read_pickle("./dataframe_test.pkl")

    y_train, x_train, y_test, x_test = preProcessData(df_train, df_test)
    #hist = trainModel(x_train, y_train)
    testModel(x_test, 53)
    #plotSetting(hist, '/home/joaos/Desktop/EST/SE/SmartPlugs/flaskApp/static/imgs/trainedModels/saved.png')
