# Face Recognition and Database Storage

This project implements a face recognition system using Python, OpenCV, and SQLite. The application captures video from a webcam, recognizes faces, and stores face encodings along with associated images in a SQLite database. The database is updated in real-time as new faces are detected or existing faces are recognized.

## Features

- **Real-time Face Detection**: Captures video from a webcam and detects faces in real-time.
- **Face Recognition**: Compares detected faces with known faces stored in the database using Euclidean distance.
- **Database Storage**: Stores face encodings, associated images, and timestamps in a SQLite database.
- **Automatic Database Updates**: Adds new faces or updates existing records in the database.
- **User Interface**: Displays the video feed with bounding boxes and labels for recognized faces.

## Prerequisites

- Python 3.x
- OpenCV
- face_recognition library
- SQLite

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/face-recognition-db.git
   cd face-recognition-db
