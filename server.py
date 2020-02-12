#!/usr/bin/env python3

import socket
import re
import json
#from _thread import start_new_thread
import threading
import os, errno, sys
import time
import matplotlib.pyplot as plt
import numpy as np
import wavePreparation
import neuralNet
import pandas as pd
import tensorflow as tf
'''
LABELS:
0 - COMPUTER
1 - LAMP
2 - COMPUTER + LAMP
3 - SCREEN
4 - COMPUTER + SCREEN
5 - LAMP + SCREEN
6 - All Together
'''

class Node:
    def __init__(self, id, addr):
        self.nodeId = id   
        self.nodeAdd = addr
        self.wave = []
        self.efValue = 0
        self.train = False
        self.label = 5
        self.welcome()
        
        
    def welcome(self):
        print(self.nodeId, self.nodeAdd)
        
        
    def shoutWave(self):
        print(self.wave)
        
        
    def saveWave(self, wave, path):

        fig = plt.figure()
        plt.xlabel('time[ms]')
        plt.ylabel('Amplitude[A]')
        plt.plot(wave)
        plt.savefig(path + str('/lastwave.png'))
        plt.close(fig)
        
        
    def appendWaveToJson(self, wave):
        with open('nodes.json', 'r') as file:
            json_data = json.load(file)
            for item in json_data:
                print(self.nodeAdd[1])
                if item['portnumber'] == self.nodeAdd[1]:
                    item['lastwave'] = list(wave)
        with open('nodes.json', 'w') as file:
            json.dump(json_data, file, indent=2)
            

class Server:
    def __init__(self ,HOST, PORT):
        self.HOST = HOST
        self.PORT = PORT
        self.nodeId = 0
        self.clientAdd = ''
        self.nodes = []
        self.train_path = './dataframe_train.pkl'
        self.test_path = './dataframe_test.pkl'
        
        
    def connectServer(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except socket.error as msg:
            print("Could not create socket. Error Code: ", str(msg.args[0]), "Error: ", msg.args[1])
            sys.exit(0)
        print("[-] Socket Created")
        # bind socket
        try:
            s.bind((self.HOST, self.PORT))
            print("[-] Socket Bound to port " + str(self.PORT))
        except socket.error as msg:
            print("Bind Failed. Error Code: {} Error: {}".format(str(msg.args[0]), msg.args[1]))
            sys.exit()
        s.listen(10)
        print('listening...')
        return s     
    
    
    def updateJsonFile(self, jsonfile, nodes):
        with open(jsonfile, "w") as write_file:
            data = []
            for node in nodes:
                data.append({ 
                "nodeid": int(node.nodeId),
                "ipaddress": str(tuple(node.nodeAdd)[0]),
                "portnumber":int(tuple(node.nodeAdd)[1]),
                "lastwave":''
                })
            json.dump(data, write_file)

    def convertStrToList(self, string):
        for ch in [',', '[', ']']:
            if ch in string:
                string = string.replace(ch,'')
            li = list(string.split(" "))
        return [float(i) for i in li]
    
    '''
Funcao de calculo de corrente em que adequirimos o valor lido pelo ADC
Its divided by the number of 2^n that n is the number of bits multiplied by 5000
'''
    def converToAmps(self, waveData, node):
        testWave = self.convertStrToList(waveData['wave'])
        node.efValue = float(max(testWave))
        node.efValue = ((node.efValue)/1023.0)*5000
        node.efValue = (((node.efValue) - 2585) / 100)*0.707
        newWave = []
        for value in testWave:   
            value = ((value)/1024.0)*5000
            value = ((value) - 2585)/100
            newWave.append(value)
        node.wave = newWave

    def prepareWave(self, waveData, node, lastwavepath):

        self.converToAmps(waveData, node)
        node.appendWaveToJson(node.wave)
        node.saveWave(node.wave, lastwavepath)
        print(node.efValue)
        wavePrep = wavePreparation.WavePrepare(node.wave)
        # plt.plot(node.wave)
        # plt.show()
        path = wavePrep.preparePath(node.train)
        wavePrep.toSpectrogram(np.asarray(node.wave), path)
        wavePrep.imgResizeGrayScale(path)
        wavePrep.addToDataSet('.', node.label , node.wave, node.efValue ,node.train ,self.train_path, self.test_path)
        if(not node.train):
            df_test = pd.read_pickle(self.test_path)
            x_test = neuralNet.reshapeArr(df_test)
            x_test = tf.reshape(x_test,(-1, 100, 100, 1))
            result = neuralNet.testModel(x_test, len(df_test)-1)
            print(result, neuralNet.switchOutput(result))

    def handleNewClient(self, conn, addr):
        conn.send(b"{Welcome to the Server. Type messages and press enter to send.\n}")
        patern = 'SE300'
        while True:
            data = conn.recv(1024)
            if data:
                new_data = data.decode('utf-8')
                print(data.decode('utf-8'))
                if(re.search(patern, new_data)):
                    data = ''
                    print('we find a node')
                    print('node ID: ' + str(self.nodeId) + '\n' + 'ip add: ' + str(addr))
                    time.sleep(5)
                    wavepath = '/home/joaos/Desktop/EST/SE/SmartPlugs/flaskApp/static/nodes/node{number}/train'.format(number = self.nodeId)
                    lastwave = '/home/joaos/Desktop/EST/SE/SmartPlugs/flaskApp/static/nodes/node{number}/lastwave'.format(number = self.nodeId)
                    try:
                        os.makedirs(wavepath)
                        os.makedirs(lastwave)
                        print('created')
                        print(os.listdir(lastwave))
                    except OSError as e:
                        if e.errno != errno.EEXIST:
                            raise
                    this_node = Node(self.nodeId, addr)
                    self.nodes.append(this_node)
                    self.nodeId = self.nodeId + 1
                    self.updateJsonFile('nodes.json', self.nodes)
                    conn.send(bytes('ack', encoding = 'utf-8'))
                    if(conn.recv(1024).decode('utf-8') == 'ack'):
                        while(True):
                            print('listening')
                            wavelen = conn.recv(4).decode('utf-8')
                            data = bytearray()
                            while len(data) < int(wavelen):
                                packet = conn.recv(int(wavelen) - len(data))
                                if not packet:
                                    return None
                                data.extend(packet)
                            waveData = json.loads(data)
                            if (this_node.train is True):
                                #prepare training model
                                threading.Thread(target=self.prepareWave(waveData, this_node, lastwave)).start()
                                print('prepare training model')
                            else:
                                threading.Thread(target=self.prepareWave(waveData, this_node, lastwave)).start()
                                print('test it')

                    break
            else:
                break
            #conn.close()
    def runServer(self, connection):
            # blocking call, waits to accept a connection
            sock = connection.accept()
            print("[-] Connected to " + sock[1][0] + ":" + str(sock[1][1]))
            target=self.handleNewClient(sock[0], sock[1])
    

s1 = Server('10.42.0.1', 11111)
open("nodes.json", "w").close()
conn = s1.connectServer()
while True:
    threading.Thread(s1.runServer(conn))
     