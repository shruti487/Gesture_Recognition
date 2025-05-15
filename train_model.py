import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# Load and combine CSVs
csv_dir = "data_collection/csv"
data = []
for file in os.listdir(csv_dir):
    df = pd.read_csv(os.path.join(csv_dir, file), header=None)
    data.append(df)
data = pd.concat(data)

X = data.iloc[:, :-1].values.astype(np.float32)
y = data.iloc[:, -1].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2)

# Train a simple model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

model.save("gesture_model.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("model_training/gesture_model.tflite", "wb") as f:
    f.write(tflite_model)

# Save label encoder
import pickle
with open("model_training/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
