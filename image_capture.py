import cv2
import os
import time

# Define gesture labels
gestures = ['thumbs_up', 'up', 'down', 'left', 'right', 'circle']

# Create folders for each gesture if not exist
for gesture in gestures:
    os.makedirs(gesture, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Function to capture images for one gesture
def capture_images_for_gesture(gesture_name):
    print(f"\n➡️ Capturing images for '{gesture_name}' gesture.")
    print("📸 Press 'c' to capture an image.")
    print("❌ Press 'n' to move to the next gesture.\n")

    image_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Show the webcam feed
        cv2.imshow(f"Capturing: {gesture_name}", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Capture image
            filename = os.path.join(gesture_name, f"{gesture_name}_{int(time.time())}.jpg")
            cv2.imwrite(filename, frame)
            image_count += 1
            print(f"✅ Captured: {filename}")
        elif key == ord('n'):  # Next gesture
            cv2.destroyWindow(f"Capturing: {gesture_name}")
            print(f"✅ Done capturing '{gesture_name}' ({image_count} images).")
            break

# Main loop to go through each gesture
for gesture in gestures:
    capture_images_for_gesture(gesture)

# Release resources after ALL gestures
cap.release()
cv2.destroyAllWindows()
print("✅ Image capture for all gestures complete.")
