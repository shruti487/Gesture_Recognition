import cv2
import csv
import os
import mediapipe as mp

label = input("Enter gesture label (e.g., swipe_left): ")
output_file = f"data/{label}.csv"
os.makedirs("data", exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # use 0 for webcam; for phone cam, use IP webcam URL

with open(output_file, mode='a', newline='') as f:
    writer = csv.writer(f)
    print(f"[INFO] Press 'c' to capture, 'q' to quit...")

    while True:
        ret, frame = cap.read()
        image = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data = []
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y, lm.z])
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                cv2.putText(image, f'Gesture: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                key = cv2.waitKey(1)
                if key == ord('c'):
                    writer.writerow(data + [label])
                    print("[INFO] Data captured.")
        cv2.imshow("Collecting Gesture", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
