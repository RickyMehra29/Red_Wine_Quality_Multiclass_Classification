import pickle
from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
scalar = pickle.load(open("Red_Wine_Quality_scaler.pkl", "rb"))
wine_model = pickle.load(open("Red_Wine_Quality_SVC.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1,-1))
    output = wine_model.predict(final_input)[0]
    print(output)
    
    return render_template('home.html', output_text="The Quality of Wine is {}.".format(output))


if __name__ == '__main__':
    app.run(debug=True)
