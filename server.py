#!/usr/bin/env python3

import socket
import re
#from _thread import start_new_thread
import threading
from os import sys
import time
import matplotlib.pyplot as plt
import numpy as np

class Node:
    def __init__(self, id, addr):
        self.nodeId = id   
        self.nodeAdd = addr
        self.welcome()
    def welcome(self):
        print(self.nodeId, self.nodeAdd)
        
        
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
                    self.nodes.append(Node(self.nodeId, addr))
                    self.nodeId = self.nodeId + 1
                    conn.send(bytes('ack', encoding = 'utf-8'))
                    if(conn.recv(1024).decode('utf-8') == 'ack'):
                        wave = []
                        wavelen = int(conn.recv(28).decode('utf-8'))
                        for i in range(0, wavelen):
                            wave.append(float(conn.recv(28).decode('utf-8')))
                            print(wave[i])
                        print(wave)
                        fs = 50
                        maxtime = 0.08
                        t = np.linspace(0, maxtime, 2 * fs, endpoint=False)
                        plt.plot(t, wave)
                        plt.show()
                        #wave = conn.recv(1024).decode('utf-8')
                        #print(wave)
                    break
            else:
                break
            #conn.close()
                #yield Node(conn, self.nodes)
            # else:
            #     reply = bytes(str('ok ...' + data.decode('utf-8')), encoding = 'utf-8')
            # conn.sendall(data)
        #conn.close() 
    def sendData(self, connection, data):
        connection[0].sendall(str(data))
        
    def recieveData(self, connection):
        return connection[0].recv(1024)
    
    def runServer(self, connection):
        while True:
            # blocking call, waits to accept a connection
            sock = connection.accept()
            print("[-] Connected to " + sock[1][0] + ":" + str(sock[1][1]))

            threading.Thread(target=self.handleNewClient(sock[0], sock[1]))
    

s1 = Server('127.0.0.1', 11111)
conn = s1.connectServer()
s1.runServer(conn)
     