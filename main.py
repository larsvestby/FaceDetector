import face_recognition
import cv2
import numpy as np
import os
import time
from scipy.spatial import distance

# Configuration
KNOWN_FACES_DIR = "faces_to_know"
ENROLLMENT_POSES = ['front', 'left', 'right', 'up', 'down']
EYE_AR_THRESHOLD = 0.25
MOUTH_AR_THRESHOLD = 0.85
RECOGNITION_THRESHOLD = 0.5


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)


def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    if not os.path.exists(KNOWN_FACES_DIR):
        return

    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if os.path.isdir(person_dir):
            valid_encodings = 0
            for filename in os.listdir(person_dir):
                image_path = os.path.join(person_dir, filename)
                try:
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        known_face_encodings.append(encodings[0])
                        known_face_names.append(person_name)
                        valid_encodings += 1
                except Exception as e:
                    print(f"Error loading {image_path}: {str(e)}")

            if valid_encodings == 0:
                print(f"Warning: No valid encodings found for {person_name}")


def enroll_new_person():
    name = input("Enter the person's name: ").strip()
    if not name:
        print("Invalid name, enrollment canceled")
        return

    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    pose_index = 0
    last_capture = 0
    enrollment_done = False
    captured_count = 0

    while not enrollment_done:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        quality_ok = False
        brightness = 0

        if face_locations:
            top, right, bottom, left = face_locations[0]
            face_ratio = ((bottom - top) * (right - left)) / (small_frame.shape[0] * small_frame.shape[1])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(gray)

            if (face_ratio > 0.1 and
                    sharpness > 100 and
                    100 < brightness < 200):
                quality_ok = True
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            cv2.rectangle(frame,
                          (left * 4, top * 4),
                          (right * 4, bottom * 4),
                          color, 2)

        cv2.putText(frame, f"Pose: {ENROLLMENT_POSES[pose_index]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Captured: {captured_count}/20 | Brightness: {brightness:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Hold still for auto-capture", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if quality_ok and (time.time() - last_capture) > 1.5:
            face_image = frame[top * 4:bottom * 4, left * 4:right * 4]
            filename = os.path.join(person_dir, f"{ENROLLMENT_POSES[pose_index]}_{captured_count}.jpg")
            cv2.imwrite(filename, face_image)
            captured_count += 1
            last_capture = time.time()
            pose_index = (pose_index + 1) % len(ENROLLMENT_POSES)

            if captured_count >= 20:
                enrollment_done = True

        cv2.imshow("Face Enrollment", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Initialize face database
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
load_known_faces()

# Main recognition loop
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Process every other frame to speed up
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame, face_locations)

    face_names = []
    # Modified recognition section in the main loop
    for (encoding, landmarks) in zip(face_encodings, face_landmarks_list):
        # Liveness check
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        mouth = landmarks['top_lip'] + landmarks['bottom_lip']

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        mar = mouth_aspect_ratio(mouth)

        liveness = "Real" if ear < EYE_AR_THRESHOLD and mar < MOUTH_AR_THRESHOLD else "Spoof"

        # Only attempt recognition if we have known faces
        if known_face_encodings:
            face_distances = face_recognition.face_distance(known_face_encodings, encoding)
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] < RECOGNITION_THRESHOLD:
                name = known_face_names[best_match_index]
            else:
                name = "Unknown"
        else:
            name = "Unknown (No registered faces)"

        face_names.append(f"{name} ({liveness})")

    # Display results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4;
        right *= 4;
        bottom *= 4;
        left *= 4
        color = (0, 255, 0) if "Real" in name else (0, 0, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    cv2.imshow('Face Recognition System', frame)

    key = cv2.waitKey(1)
    if key == ord('a'):
        video_capture.release()
        enroll_new_person()
        video_capture = cv2.VideoCapture(0)
        load_known_faces()
    elif key in [ord('q'), 27]:  # 27 = ESC
        break

video_capture.release()
cv2.destroyAllWindows()