from flask import Flask, request

app = Flask(__name__)

@app.route("/hello/<val>")
def hello_world(val):
    return "<p>Hello, World!</p>" + val


@app.route("/sum/<x>/<y>")
def sum_num(x,y):
    sum = int(x) + int(y)
    return str(sum)


@app.route("/model", methods = ['POST'])
def pred_model():
    js = request.get_json()
    x = js['x']
    y = js['y']
    sum = int(x) + int(y)
    return str(sum)
