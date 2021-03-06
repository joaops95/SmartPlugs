import socket
import numpy as np 
import random
import matplotlib.pyplot as plt
import sys
import time
import struct
import json
import librosa
import librosa.display

with open('./nodes.json') as json_file:
    data = json.load(json_file)
    for p in data:
        nodeid = p['nodeid']
        wave = np.asarray(p['lastwave'])

# HOST = '127.0.0.1'  # The server's hostname or IP address
# PORT = 11111        # The port used by the server
# fs = 50
# maxtime = 0.08
# numSamples = (maxtime/(1/fs))
# t = np.linspace(0, maxtime, 2 * fs, endpoint=False)
# pure = random.randrange(2,5)*np.sin(fs * 2 * np.pi * t)
# # print(pure)
# # plt.plot(t, pure)
# # plt.show()
# info = ['']
# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     s.connect((HOST, PORT))
#     data = s.recv(1024)
#     print('Received', repr(data.decode("utf-8")))
#     s.sendall(b'SE300')
#     data = s.recv(10240)
#     if (data.decode('utf-8') == 'ack'):
#         s.send(bytes('ack', encoding = 'utf-8'))
#         print('fecha o rele')
#         print('envia info da onda')
#         print(sys.getsizeof(len(pure)))
#         s.send(bytes(str(len(pure)), encoding = 'utf-8'))
#         buf = struct.pack('%sf' % len(pure), *pure)
#         s.sendall(buf)
#         print('sent!!')