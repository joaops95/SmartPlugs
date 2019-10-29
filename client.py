#!/usr/bin/env python3

import socket
import numpy as np 
import random
import matplotlib.pyplot as plt
import sys
import time

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 11111        # The port used by the server
fs = 50
maxtime = 0.08
numSamples = (maxtime/(1/fs))
t = np.linspace(0, maxtime, 2 * fs, endpoint=False)

pure = random.randrange(2,5)*np.sin(fs * 2 * np.pi * t)
# print(pure)
# plt.plot(t, pure)
# plt.show()
info = ['']

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    data = s.recv(1024)
    print('Received', repr(data.decode("utf-8")))
    s.sendall(b'SE300')
    data = s.recv(1024)
    if (data.decode('utf-8') == 'ack'):
        s.send(bytes('ack', encoding = 'utf-8'))
        print('fecha o rele')
        print('envia info da onda')
        print(sys.getsizeof(len(pure)))
        s.send(bytes(str(len(pure)), encoding = 'utf-8'))
        #print(sys.getsizeof(s.sendall(bytes(pure))))
        for element in pure:
            s.sendall(bytes(str(round(element, 4)), encoding= 'utf=8'))
            time.sleep(0.03)
            #print(sys.getsizeof(s.send(bytes(str(round(element, 3)), encoding= 'utf=8'))))
        print('sent!!')
