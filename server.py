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
import struct

class Node:
    def __init__(self, id, addr):
        self.nodeId = id   
        self.nodeAdd = addr
        self.wave = []
        self.efValue = 0
        self.train = True
        self.trainCat = 0
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
    
    def prepareWave(self, waveData, node, lastwavepath):
        node.wave = self.convertStrToList(waveData['wave'])
        node.efValue = float(waveData['efValue'])
        node.appendWaveToJson(node.wave)
        node.saveWave(node.wave, lastwavepath)
        wavePrep = wavePreparation.WavePrepare(node.wave)
        path = wavePrep.preparePath()
        wavePrep.toSpectrogram(node.wave, path)
        wavePrep.imgResizeGrayScale(path)
        
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
                    #conn.send(bytes('SE300-Match', encoding='utf-8'))
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
                            wavelen = conn.recv(4).decode('utf-8')
                            print(wavelen)
                            data = bytearray()
                            while len(data) < int(wavelen):
                                packet = conn.recv(int(wavelen) - len(data))
                                if not packet:
                                    return None
                                data.extend(packet)
                            waveData = json.loads(data)
                            if (this_node.train is True):
                                #prepare training model
                                self.prepareWave(waveData, this_node, lastwave)
                                print('prepare training model')
                            else:
                                print('test it')

                            plt.plot(this_node.wave)
                            plt.show()

                    break
            else:
                break
            #conn.close()


    def runServer(self, connection):
        while True:
            # blocking call, waits to accept a connection
            sock = connection.accept()
            print("[-] Connected to " + sock[1][0] + ":" + str(sock[1][1]))

            threading.Thread(target=self.handleNewClient(sock[0], sock[1]))
    

s1 = Server('10.42.0.1', 11111)
open("nodes.json", "w").close()
conn = s1.connectServer()
s1.runServer(conn)
     