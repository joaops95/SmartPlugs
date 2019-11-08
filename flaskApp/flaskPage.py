from flask import Flask, render_template, url_for
import json

app = Flask(__name__)
def getNodes():
    with open("../nodes.json", "r") as read_file:
        return json.load(read_file)
    
@app.route('/')
@app.route('/home')
def hello():
    return render_template('home.html', nodes=getNodes())


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=True)