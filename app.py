from flask import Flask, render_template, request
import tensorflow as tf
from joblib import load
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load data and preprocessing objects
df = pd.read_csv('yield_df.csv')
model = tf.keras.models.load_model('model.h5')
label_encoder = load('label_encoder.joblib')
scaler = load('scaler.joblib')

@app.route('/')
def home():
    return render_template('chatbot.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['userInput']

    # Check if user input is in the categories the label_encoder was trained on
    if user_input not in label_encoder.classes_:
        # Handle new category
        print(f'Unknown category: {user_input}')
        return 'Unknown category. Please try again with a known category.'

    # Process user input to get future crop predictions for the chosen vegetable
    future_data = np.array([[2023, user_input]])

    # Encode the categorical feature
    future_data_encoded = future_data.copy()
    future_data_encoded[:, 1] = label_encoder.transform(future_data_encoded[:, 1])

    # Convert data to TensorFlow tensors
    future_data_encoded = tf.convert_to_tensor(future_data_encoded, dtype=tf.float32)

    # Make predictions for the chosen vegetable
    predictions = model.predict(future_data_encoded)

    # Denormalize the predictions
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    # Prepare the results as a list of dictionaries
    crop_predictions = [
        {'crop': user_input, 'year': int(year), 'yield': yield_amount}
        for year, yield_amount in zip(future_data[:, 0], predictions)
    ]

    return render_template('results.html', crop_predictions=crop_predictions)

if __name__ == '__main__':
    app.run(debug=True)
