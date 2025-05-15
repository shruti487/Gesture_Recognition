import os
import cv2
import mediapipe as mp
import csv

DATA_DIR = "data_collection/data"
CSV_DIR = "data_collection/csv"
os.makedirs(CSV_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0]
        return [coord for lm in landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
    return None

for label in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_path): continue

    output_csv = os.path.join(CSV_DIR, f"{label}.csv")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            image = cv2.imread(img_path)
            if image is None: continue
            landmarks = extract_landmarks(image)
            if landmarks:
                writer.writerow(landmarks + [label])
    print(f"Processed {label}")
