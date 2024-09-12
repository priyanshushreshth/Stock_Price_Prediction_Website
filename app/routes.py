from flask import render_template, request
from app import app
import pickle

# Load the trained model
with open('model/model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['data']
    # Process the data and make prediction
    # Example: Convert to appropriate format if needed
    prediction = model.predict([data])  # Adjust as needed
    return render_template('results.html', prediction=prediction)
