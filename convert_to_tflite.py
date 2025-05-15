import tensorflow as tf
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and merge all CSVs
data_path = 'data_collection/data'
all_data = []

# Loop through CSV files and load the data
for file in os.listdir(data_path):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(data_path, file), header=None)
        all_data.append(df)

# Concatenate all the data into a single DataFrame
data = pd.concat(all_data, ignore_index=True)

# Separate features (X) and labels (y)
X = data.iloc[:, :-1].values  # All columns except the last one (features)
y = data.iloc[:, -1].values  # The last column (label)

# Ensure all values in X are numeric (convert non-numeric to NaN for each column)
X = pd.DataFrame(X)  # Convert to DataFrame to use pandas functions
X = X.apply(pd.to_numeric, errors='coerce')  # Apply pd.to_numeric column-wise

# Replace NaN values with 0 (or use other imputation techniques as necessary)
X = X.fillna(0).values  # Converts NaN to 0

# Convert data to numpy arrays of appropriate data types
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=str)  # Use Python's built-in string type

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(y_cat.shape[1], activation='softmax')  # Output layer with number of categories
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# Save the trained model
model.save("gesture_model.h5")

# Convert to TFLite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open("gesture_model.tflite", "wb") as f:
    f.write(tflite_model)

print("[INFO] Model trained, saved as gesture_model.h5 and gesture_model.tflite")
