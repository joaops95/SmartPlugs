# import json
# import os
# import sys
# sys.path.append('/home/joaos/Desktop/EST/SE/SmartPlugs')
# import testes


# def getLastWaveOfNode():
#     with open("../nodes.json", "r") as read_file:
#         data = json.load(read_file)
#         pathsOfLastWaves = []
#         for node in data:
#                 dr = './static/nodes/node{nodeId}/lastwave'.format(nodeId = node['nodeid'])
#                 lastf = os.listdir(dr)[0]
#                 pathsOfLastWaves.append(dr + '/' + lastf)
#         return pathsOfLastWaves


# pfrint()

# print(getLastWaveOfNode())

    for i in range(1:10):
        error = 512 - analogRead(readValue)
        sensorZeroAdj = ((sensorZeroAdj * (i-1)) + error)/i