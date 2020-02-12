#from PyAccessPoint import pyaccesspoint
import json
import os
# class Hotspot:
#     def __init__(self):
#         self.ssid = 'SmartHome'
#         self.password = 'seproject'
#         self.wlan = 'wlp3s0'
#         self.inet = None
#         self.ip = '192.168.0.1'
#         self.netMask = '255.255.255.0'
#         self.running = False
#         self.access_point = pyaccesspoint.AccessPoint(wlan=self.wlan, inet= self.inet, ip=self.ip, netmask= self.netMask,ssid=self.ssid, password=self.password)
        
#     def openHotspot(self):
#         self.access_point.start()
#         self.running = self.access_point.is_running()
    
#     def closeHotspot(self):
#         if(self.running):
#             self.access_point.stop()
#             self.running = self.access_point.is_running()
#         else:
#             pass
'''
This code doesnt work for ubunut 18... lets hack :)
'''
# loaded_json = json.loads( '/etc/accesspoint/accesspoint.json')
# print(loaded_json)

class Hotspot:
    
    def __init__(self, jsonPath):
        os.system('sudo chmod +777 ' + str(jsonPath))
        with open(jsonPath, "r") as jsonRead:
            jsonData = json.loads(jsonRead.read())
        self.ssid = jsonData['ssid']
        self.password = jsonData['password']
        self.wlan = jsonData['wlan']
        self.inet = jsonData['inet']
        self.ip = jsonData['ip']
        self.netmask = jsonData['netmask']
        self.running = False
    
    def openAP(self):
        startapcommand = 'sudo pyaccesspoint --config start'
        os.system(startapcommand)
        self.running = True
        print(self.ssid, self.password, self.wlan, self.inet, self.ip, self.netmask)

    def closeAP(self):
        stopapcommand = 'sudo pyaccesspoint --config stop'
        os.system(stopapcommand)
        self.running = False
h1 = Hotspot('/etc/accesspoint/accesspoint.json')
h1.closeAP()