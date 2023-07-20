import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from joblib import dump

# Load the dataset from a CSV file
df = pd.read_csv('yield_df.csv')

# Extract the input features and target variable
X = df[['Year', 'Item']].values
y = df['hg/ha_yield'].values

# Encode categorical features
label_encoder = LabelEncoder()
X[:, 1] = label_encoder.fit_transform(X[:, 1])

# Normalize the yield amount
scaler = MinMaxScaler()
y = scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Split data into training and testing sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Convert data to TensorFlow tensors
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

# Define the model architecture
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=2))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Evaluate the model
loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {loss}')

# Save the trained model
model.save('model.h5')

# Save the label encoder and scaler
dump(label_encoder, 'label_encoder.joblib')
dump(scaler, 'scaler.joblib')

print("Model, label encoder and scaler have been saved.")
