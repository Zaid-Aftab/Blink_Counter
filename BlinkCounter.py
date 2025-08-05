import cv2
import dlib
import numpy as np

from scipy.spatial import distance as dist

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file!

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])  # Vertical distance
    B = dist.euclidean(eye[2], eye[4])  # Vertical distance
    C = dist.euclidean(eye[0], eye[3])  # Horizontal distance
    return (A + B) / (2.0 * C)  # EAR formula

# Eye landmark indices
LEFT_EYE_IDX = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 48))

# Blink threshold & counter
EAR_THRESHOLD = 0.22  # Adjust if needed
BLINK_CONSEC_FRAMES = 2  
blink_count = 0
frame_counter = 0

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Error: Couldn't open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Error: Cannot read frame.")
        break

    frame = cv2.flip(frame, 1)  # Fix mirrored effect
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Get eye landmarks
        left_eye_pts = [ (landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_IDX ]
        right_eye_pts = [ (landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_IDX ]

        # Calculate EAR
        left_EAR = eye_aspect_ratio(left_eye_pts)
        right_EAR = eye_aspect_ratio(right_eye_pts)
        avg_EAR = (left_EAR + right_EAR) / 2.0  # Average EAR of both eyes

        # Draw eyes
        for eye in [left_eye_pts, right_eye_pts]:
            x, y = min(p[0] for p in eye), min(p[1] for p in eye)
            w, h = max(p[0] for p in eye) - x, max(p[1] for p in eye) - y
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Blink detection
        if avg_EAR < EAR_THRESHOLD:
            frame_counter += 1
        else:
            if frame_counter >= BLINK_CONSEC_FRAMES:
                blink_count += 1
            frame_counter = 0  # Reset frame counter

    # Display blink count
    cv2.putText(frame, f"Blinks: {blink_count}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Blink Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        print(blink_count)
        break

cap.release()
cv2.destroyAllWindows()
