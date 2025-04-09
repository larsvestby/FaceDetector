import face_recognition
import cv2
import numpy as np
import os
import time
import pickle
from scipy.spatial import distance
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor

# Configuration
FACE_DETECTION_MODEL = 'hog'
KNOWN_FACES_DIR = "faces_to_know"
CACHE_FILE = "known_faces_cache.pkl"
ENROLLMENT_POSES = ['front', 'left', 'right', 'up', 'down']
EYE_AR_THRESHOLD = 0.25
MOUTH_AR_THRESHOLD = 0.85
RECOGNITION_THRESHOLD = 0.5
PROCESS_INTERVAL = 1
RESIZE_FACTOR = 0.5

latest_face_info = []
face_info_lock = Lock()
encoding_executor = ThreadPoolExecutor(max_workers=1)

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

def save_known_faces(known_face_encodings, known_face_names):
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({
            "encodings": known_face_encodings,
            "names": known_face_names
        }, f)

def build_known_faces():
    known_face_encodings = []
    known_face_names = []

    if not os.path.exists(KNOWN_FACES_DIR):
        return known_face_encodings, known_face_names

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
    return known_face_encodings, known_face_names

def load_known_faces(force_rebuild=False):
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    # If force_rebuild is True, ignore the cache and rebuild it.
    if not force_rebuild and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                data = pickle.load(f)
                known_face_encodings = data.get("encodings", [])
                known_face_names = data.get("names", [])
            print("Loaded known faces from cache.")
            return
        except Exception as e:
            print(f"Error loading cache: {e}. Rebuilding cache...")

    # Rebuild the cache from the directory
    known_face_encodings, known_face_names = build_known_faces()
    save_known_faces(known_face_encodings, known_face_names)
    print("Saved known faces to cache.")

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
    # Rebuild the cache after enrolling a new person (force a rebuild)
    load_known_faces(force_rebuild=True)

def process_encodings(rgb_small_frame, face_locations, known_encodings, known_names):
    """Heavy processing done in a separate thread"""
    face_encodings = face_recognition.face_encodings(
        rgb_small_frame,
        face_locations,
        num_jitters=1  # Reduced from 3
    )

    face_landmarks_list = face_recognition.face_landmarks(
        rgb_small_frame,
        face_locations
    )

    face_info = []
    for (encoding, landmarks) in zip(face_encodings, face_landmarks_list):
        # Calculate liveness
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        mouth = landmarks['top_lip'] + landmarks['bottom_lip']
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        mar = mouth_aspect_ratio(mouth)
        liveness = "Real" if ear < EYE_AR_THRESHOLD and mar < MOUTH_AR_THRESHOLD else "Spoof"

        # Calculate recognition
        name = "Unknown"
        confidence = None
        if known_encodings:
            face_distances = face_recognition.face_distance(known_encodings, encoding)
            best_match_index = np.argmin(face_distances)
            min_distance = face_distances[best_match_index]

            if min_distance < RECOGNITION_THRESHOLD:
                name = known_names[best_match_index]
                confidence = round((1 - min_distance) * 100)

        face_info.append({
            'name': name,
            'confidence': confidence,
            'liveness': liveness
        })

    return face_info

def recognition_thread(video_capture):
    global latest_face_info

    last_processed = 0
    current_face_locations = []
    future = None
    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            # Always detect face locations (fast)
            small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            current_face_locations = face_recognition.face_locations(
                rgb_small_frame,
                model=FACE_DETECTION_MODEL
            )

            # Trigger full processing at intervals
            if time.time() - last_processed > PROCESS_INTERVAL:
                if future is None or future.done():
                    # Submit new processing task
                    future = encoding_executor.submit(
                        process_encodings,
                        rgb_small_frame,
                        current_face_locations,
                        known_face_encodings,
                        known_face_names
                    )
                    last_processed = time.time()

            # Update face info if available
            if future and future.done():
                with face_info_lock:
                    latest_face_info = future.result()
                future = None

            # Draw real-time face boxes
            for (top, right, bottom, left) in current_face_locations:
                # Scale back up face locations
                top = int(top * (1 / RESIZE_FACTOR))
                right = int(right * (1 / RESIZE_FACTOR))
                bottom = int(bottom * (1 / RESIZE_FACTOR))
                left = int(left * (1 / RESIZE_FACTOR))

                # Draw basic box (will be updated when info arrives)
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw recognition info when available
            with face_info_lock:
                for (loc, info) in zip(current_face_locations, latest_face_info):
                    top, right, bottom, left = loc
                    top = int(top * (1 / RESIZE_FACTOR))
                    right = int(right * (1 / RESIZE_FACTOR))
                    bottom = int(bottom * (1 / RESIZE_FACTOR))
                    left = int(left * (1 / RESIZE_FACTOR))

                    color = (0, 255, 0) if info['liveness'] == "Real" else (0, 0, 255)
                    label = f"{info['name']} {info['confidence']}%" if info['confidence'] else info['name']

                    # Update boxes with latest info
                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                    cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                    cv2.putText(display_frame, label, (left + 6, bottom - 6),
                                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            cv2.imshow('Face Recognition System', display_frame)

            key = cv2.waitKey(1)
            if key == ord('a'):
                video_capture.release()
                enroll_new_person()
                video_capture = cv2.VideoCapture(0)
                # Force a rebuild of the cache after enrollment to update changes immediately
                load_known_faces(force_rebuild=True)
            elif key in [ord('q'), 27]:
                break
    finally:
        video_capture.release()
        cv2.destroyAllWindows()
        #encoding_executor.shutdown()

if __name__ == "__main__":
    try:
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        load_known_faces(force_rebuild=True)
        video_capture = cv2.VideoCapture(0)
        recognition_thread(video_capture)
    finally:
        encoding_executor.shutdown()
