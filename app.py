
import flask
from main import predict_handler


app = flask.Flask("CloudFunction")


@app.route("/", methods=["GET", "POST"])
def predict():
    return predict_handler()


app.run() 
