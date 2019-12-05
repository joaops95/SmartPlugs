from flask import Flask, render_template, url_for
import json
import os

app = Flask(__name__)
def getNodes():
    with open("../nodes.json", "r") as read_file:
        return json.load(read_file)

def getLastWaveOfNode():
    with open("../nodes.json", "r") as read_file:
        data = json.load(read_file)
        pathsOfLastWaves = []
        for node in data:
                dr = './static/nodes/node{nodeId}/lastwave'.format(nodeId = node['nodeid'])
                lastf = os.listdir(dr)[0]
                pathsOfLastWaves.append(dr + '/' + lastf)
        return pathsOfLastWaves
        
@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html', nodes=getNodes())


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/houseConsumption')
def houseConsumption():
        return render_template('houseconsumption.html')

@app.route('/systemaccuracy')
def systemaccuracy():
        return render_template('systemaccuracy.html')

@app.route('/allequipments')
def allequipments():
        return render_template('allequipments.html', wavesPath = getLastWaveOfNode())

@app.route('/settings')
def settings():
        return render_template('settings.html')


if __name__ == "__main__":
    app.run(debug=True)