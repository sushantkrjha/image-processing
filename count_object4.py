import face_recognition
import cv2
import sqlite3
from datetime import datetime
import numpy as np
from scipy.spatial import distance

def init_db():
    conn = sqlite3.connect('people.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT NOT NULL,
            face_encoding BLOB,
            image BLOB,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    return conn

def save_or_update_record(conn, person_id, face_encoding, image, timestamp):
    #print(face_encoding)
    c = conn.cursor()
    
    # Check if person already exists
    c.execute('''
        SELECT id FROM people WHERE person_id = ?
    ''', (person_id,))
    
    if c.fetchone():
        # Update existing record
        c.execute('''
            UPDATE people SET face_encoding = ?, image = ?, timestamp = ? WHERE person_id = ?
        ''', (face_encoding, image, timestamp, person_id))
    else:
        # Insert new record
        c.execute('''
            INSERT INTO people (person_id, face_encoding, image, timestamp) VALUES (?, ?, ?, ?)
        ''', (person_id, face_encoding, image, timestamp))
    
    conn.commit()

def fetch_known_face_encodings(conn):
    c = conn.cursor()
    c.execute('SELECT person_id, face_encoding FROM people')
    rows = c.fetchall()
    known_face_encodings = []
    known_face_ids = {}
    for row in rows:
        person_id, face_encoding_blob = row
        face_encoding = np.frombuffer(face_encoding_blob, dtype=np.float64)
        known_face_encodings.append(face_encoding)
        known_face_ids[person_id] = face_encoding
    return known_face_encodings, known_face_ids

def process_frame(frame, known_face_encodings, known_face_ids, conn):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face with known faces
        distances = [distance.euclidean(face_encoding, known_face) for known_face in known_face_encodings]
        min_distance = min(distances) if distances else float('inf')
        tolerance = 0.6
        name = "Unknown"

        if min_distance < tolerance:
            match_index = distances.index(min_distance)
            name = list(known_face_ids.keys())[match_index]
        else:
            name = f"Person_{len(known_face_encodings)}"
            known_face_encodings.append(face_encoding)
            known_face_ids[name] = face_encoding
        
        # Extract the face image
        face_image = frame[top:bottom, left:right]
        _, img_encoded = cv2.imencode('.jpg', face_image)
        img_bytes = img_encoded.tobytes()

        # Save or update the database record
        timestamp = datetime.now().isoformat()
        save_or_update_record(conn, name, face_encoding.tobytes(), img_bytes, timestamp)
        
        # Draw bounding boxes
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame, known_face_encodings, known_face_ids

def count_obj():
    cap = cv2.VideoCapture(0)  # For webcam, use '0'
    conn = init_db()
    known_face_encodings, known_face_ids = fetch_known_face_encodings(conn)

    if not cap.isOpened():
        print("Error: Could not open video source.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame to extract and save person images
        frame, known_face_encodings, known_face_ids = process_frame(frame, known_face_encodings, known_face_ids, conn)

        # Display the frame
        cv2.imshow('Video Feed', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()

if __name__ == "__main__":
    count_obj()
