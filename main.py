import face_recognition
import cv2
import numpy as np
import os
import time
from scipy.spatial import distance
from threading import Thread

# Configuration
FACE_DETECTION_MODEL = 'hog'
KNOWN_FACES_DIR = "faces_to_know"
ENROLLMENT_POSES = ['front', 'left', 'right', 'up', 'down']
EYE_AR_THRESHOLD = 0.25
MOUTH_AR_THRESHOLD = 0.85
RECOGNITION_THRESHOLD = 0.5

def eye_aspect_ratio(eye):
    a = distance.euclidean(eye[1], eye[5])
    b = distance.euclidean(eye[2], eye[4])
    c = distance.euclidean(eye[0], eye[3])
    return (a + b) / (2.0 * c)

def mouth_aspect_ratio(mouth):
    a = distance.euclidean(mouth[2], mouth[10])
    b = distance.euclidean(mouth[4], mouth[8])
    c = distance.euclidean(mouth[0], mouth[6])
    return (a + b) / (2.0 * c)

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
                    face_locations = face_recognition.face_locations(
                        image,
                        model=FACE_DETECTION_MODEL
                    )
                    encodings = face_recognition.face_encodings(
                        image,
                        face_locations,
                        num_jitters=5
                    )
                    if encodings:
                        known_face_encodings.append(encodings[0])
                        known_face_names.append(person_name)
                        valid_encodings += 1
                except Exception as e:
                    print(f"Error loading {image_path}: {str(e)}")

            if valid_encodings == 0:
                print(f"Warning: No valid encodings found for {person_name}")

def enroll_new_person():
    global top, bottom, left, right
    name = input("Enter the person's name: ").strip()
    if not name:
        print("Invalid name, enrollment canceled")
        return

    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    if not os.path.exists(person_dir):
        print("This person is not enrolled yet. Starting new enrollment.")
        os.makedirs(person_dir, exist_ok=True)

    existing_images = [f for f in os.listdir(person_dir) if f.endswith('.jpg')]
    captured_count = len(existing_images)
    new_images_to_capture = 10

    cap = cv2.VideoCapture(0)
    pose_index = 0
    last_capture = 0

    while new_images_to_capture > 0:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(
            rgb_small_frame,
            model=FACE_DETECTION_MODEL
        )
        quality_ok = False
        brightness = 0

        if face_locations:
            top, right, bottom, left = face_locations[0]
            face_ratio = ((bottom - top) * (right - left)) / (small_frame.shape[0] * small_frame.shape[1])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(gray)

            if face_ratio > 0.1 and sharpness > 100 and 100 < brightness < 200:
                quality_ok = True
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            cv2.rectangle(frame, (left * 4, top * 4), (right * 4, bottom * 4), color, 2)

        cv2.putText(frame, f"Pose: {ENROLLMENT_POSES[pose_index]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Captured: {captured_count} | Brightness: {brightness:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Remaining: {new_images_to_capture}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if quality_ok and (time.time() - last_capture) > 1.5:
            face_image = frame[top * 4:bottom * 4, left * 4:right * 4]
            filename = os.path.join(person_dir, f"{ENROLLMENT_POSES[pose_index]}_{captured_count}.jpg")
            cv2.imwrite(filename, face_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            captured_count += 1
            new_images_to_capture -= 1
            last_capture = time.time()
            pose_index = (pose_index + 1) % len(ENROLLMENT_POSES)

        cv2.imshow("Face Enrollment", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Added 10 new images for {name}.")

def process_frame(frame, known_face_encodings, known_face_names):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(
        rgb_small_frame,
        model=FACE_DETECTION_MODEL
    )
    face_encodings = face_recognition.face_encodings(
        rgb_small_frame,
        face_locations,
        num_jitters=3
    )
    face_landmarks_list = face_recognition.face_landmarks(
        rgb_small_frame,
        face_locations
    )

    face_info = []
    for (encoding, landmarks) in zip(face_encodings, face_landmarks_list):
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        mouth = landmarks['top_lip'] + landmarks['bottom_lip']

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        mar = mouth_aspect_ratio(mouth)
        liveness = "Real" if ear < EYE_AR_THRESHOLD and mar < MOUTH_AR_THRESHOLD else "Spoof"

        name = "Unknown"
        confidence = None

        if known_face_encodings:
            face_distances = face_recognition.face_distance(known_face_encodings, encoding)
            best_match_index = np.argmin(face_distances)
            min_distance = face_distances[best_match_index]

            if min_distance < RECOGNITION_THRESHOLD:
                name = known_face_names[best_match_index]
                confidence = round((1 - min_distance) * 100)
            else:
                name = "Unknown"
        else:
            name = "Unknown (No registered faces)"

        face_info.append({
            'name': name,
            'confidence': confidence,
            'liveness': liveness
        })

    return face_locations, face_info

def recognition_thread(video_capture, known_face_encodings, known_face_names):
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        face_locations, face_info = process_frame(frame, known_face_encodings, known_face_names)

        # Draw face boxes and labels
        for (top, right, bottom, left), info in zip(face_locations, face_info):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            color = (0, 255, 0) if info['liveness'] == "Real" else (0, 0, 255)

            # Create label text
            if info['confidence'] is not None:
                label = f"{info['name']} {info['confidence']}%"
            else:
                label = info['name']

            # Draw bounding box and label
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # Display liveness status in corner
        if face_info:
            liveness_statuses = [f"{info['liveness']}" for info in face_info]
            liveness_text = "Status: " + ", ".join(liveness_statuses)
            cv2.putText(frame, liveness_text, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Face Recognition System', frame)

        key = cv2.waitKey(1)
        if key == ord('a'):
            video_capture.release()
            enroll_new_person()
            video_capture = cv2.VideoCapture(0)
            load_known_faces()
        elif key in [ord('q'), 27]:
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Initialize face database
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
load_known_faces()

# Main recognition loop
video_capture = cv2.VideoCapture(0)
recognition_thread = Thread(target=recognition_thread, args=(video_capture, known_face_encodings, known_face_names))
recognition_thread.start()