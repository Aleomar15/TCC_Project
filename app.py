import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__, static_url_path='/static')
model = pickle.load(open('models/model.pkl', 'rb'))

@app.route('/')
def disply_gui():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()] #Convert string inputs to float.
    features = [np.array(int_features)]  #Convert to the form [[a, b]] for input to the model
    prediction = model.predict(features)  # features Must be in the form [[a, b]]


    return render_template('index.html', prediction_text='Quantidade de casos previstas é{}'.format(prediction))

if __name__ == "__main__":
    app.run()