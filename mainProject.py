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


def generateWaves(df_train, df_test ,t, f, fs): # this function generates 3 waves
    testEx = False
    arr1 = [] 
    arr2 = [] 
    arr3 = []
    imgindex = 0 
    imgindex1 = 0
    wavesnumbers = 500
    testImgsPercentile = 0.01
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
            df_test.loc[imgindex1] = [0, np.asarray(normalizeData(path1))]
        else:    
            df_train.loc[imgindex] = [0, np.asarray(normalizeData(path1))]
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
        imgResizeGrayScale(path1)
        if(testEx):
            df_test.loc[imgindex1] = [1, np.asarray(normalizeData(path1))]
        else:    
            df_train.loc[imgindex] = [1, np.asarray(normalizeData(path1))]
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
            df_test.loc[imgindex1] = [2, np.asarray(normalizeData(path1))]
        else:    
            df_train.loc[imgindex] = [2, np.asarray(normalizeData(path1))]
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
    plt.axis('off')
    ax.set_position([0, 0, 1, 1])
    plt.savefig(path)
    plt.close()
    
def imgResizeGrayScale(path):
    img = cv2.imread(path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (1, 250))
    cv2.imwrite(path,resized_image)
    print('img {path} resized'.format(path = path))
    
def normalizeData(path):
    normalizedArr = []
    unnormalized_arr = cv2.imread(path)
    for element in unnormalized_arr:
        normalized_value = np.mean(element) / 255.0
        #round(np.mean(element)/(255),3)
        normalizedArr.append(normalized_value)
    return normalizedArr
        
def neuralNet(df, checkpoint_path):
    x_train = df['x_train']
    y_train = df['y_train']
    x, y = preprocess(x_train, y_train)
    train_dataset = create_dataset(x, y)
    model = createModel(x, y)
    #Callbacks save model
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True,
                                                verbose=1,
                                                period=2)
    model.save_weights(checkpoint_path.format(epoch=0))
    
    history = model.fit(
        train_dataset.repeat(), 
        epochs=10, 
        steps_per_epoch=250,
        callbacks=[cp_callback]
    )
    plotSetting(history)

def plotSetting(history):
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
    plt.show()
    
def createModel(x, y):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Reshape((x.shape[1], 1), input_shape=(x.shape[1], )))
    model.add(tf.keras.layers.Conv1D(250, 10, activation='relu', input_shape=(x.shape[1], )))
    model.add(tf.keras.layers.Conv1D(250, 10, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(3))
    model.add(tf.keras.layers.Conv1D(310, 10, activation='relu'))
    model.add(tf.keras.layers.Conv1D(310, 10, activation='relu'))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.6)
    model.compile(optimizer=sgd,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model
def preprocess(x, y):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.int64)
    return x, y

def create_dataset(xs, ys, n_classes=3):
  ys = tf.one_hot(ys, depth=n_classes)
  return tf.data.Dataset.from_tensor_slices((xs, ys)) \
    .map(preprocess) \
    .shuffle(len(ys)) \
    .batch(1)

def saveWeights(filename):
    # serialize model to JSON
    checkpoint_path = str(filename) + '.ckpt'

    return checkpoint_path
    
    # later...
def loadWeights(filename):
    # load json and create model
    loaded_model = load_model(str(filename) + '.ckpt')
    print("Loaded model from disk")
    return loaded_model


if __name__ == "__main__":
    f = random.uniform(49.5,50.5)  # Frequency, in cycles per second, or Hertz
    df_train = pd.DataFrame(columns=['y_train', 'x_train'])
    df_test = pd.DataFrame(columns=['y_test', 'x_test'])
    #f = 50
    fs = 50  # Sampling rate, or number of measurements per second
    maxtime = 0.08
    numSamples = (maxtime/(1/f))
    t = np.linspace(0, maxtime, 2 * fs, endpoint=False)
    checkpoint_path = "./training_1/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    #arr1, arr2, arr3 = generateWaves(df_train,df_test ,t, f, fs)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    df_train = pd.read_pickle("./dataframe_train.pkl")
    df_test = pd.read_pickle("./dataframe_test.pkl")
    
    #neuralNet(df_train, checkpoint_path)
    
    xtrain, ytrain = preprocess(df_train['x_train'], df_train['y_train'])
    xtest, ytest = preprocess(df_test['x_test'],  df_test['y_test'])

    model = createModel(xtrain, ytrain) 
    print(latest)
    model.load_weights(latest)
    loss, acc = model.evaluate(xtrain, ytrain , verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))
    ynew = model.predict_classes(np.transpose(df_test['x_test'][1]))
    print(ynew, df_test['y_test'][1])

    # print(df_train.head())
    # print(df_test)