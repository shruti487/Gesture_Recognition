import tensorflow as tf
import numpy as np
import os
import pickle

model_path = os.path.join(os.path.dirname(__file__), '..', 'model_training', 'gesture_model.tflite')
label_path = os.path.join(os.path.dirname(__file__), '..', 'model_training', 'label_encoder.pkl')

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(label_path, 'rb') as f:
    label_encoder = pickle.load(f)

def predict_landmarks(landmarks):
    input_array = np.array(landmarks, dtype=np.float32).reshape(1, -1)
    interpreter.set_tensor(input_details[0]['index'], input_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return label_encoder.inverse_transform([np.argmax(output)])[0]
