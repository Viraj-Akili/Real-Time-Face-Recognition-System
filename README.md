# Real-Time Face Recognition System

## Overview

A CPU-optimized real-time face recognition system that performs face detection using OpenCV DNN and identity recognition using facial embeddings generated with `face_recognition`.

The system supports real-time webcam recognition, face tracking, and duplicate prevention to improve efficiency and reduce unnecessary recognition operations.

## Features

* Real-time webcam face detection
* Face recognition using facial embeddings
* OpenCV DNN (ResNet SSD) face detector
* Face tracking using OpenCV trackers
* Duplicate prevention via tracking
* CPU-optimized processing pipeline
* Modular and easy-to-extend architecture

## Tech Stack

* Python 3.11
* OpenCV (DNN + Tracking)
* face_recognition
* dlib
* NumPy

## Project Structure

```text
Face-Detection/
│
├── known_faces/
├── encode_known_faces.py
├── realtime_face_recog.py
├── known_faces.pkl
├── deploy.prototxt
├── res10_300x300_ssd_iter_140000.caffemodel
├── requirements.txt
└── README.md
```

## How It Works

1. Detect faces using OpenCV DNN (ResNet SSD).
2. Extract 128-dimensional facial embeddings.
3. Compare embeddings against stored known-face encodings.
4. Assign the closest matching identity.
5. Track detected faces to reduce duplicate recognition operations.

## Installation

Clone the repository:

```bash
git clone https://github.com/Viraj-Akili/Real-Time-Face-Recognition-System.git
cd Real-Time-Face-Recognition-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Encode Known Faces

Place images inside the `known_faces` directory and run:

```bash
python encode_known_faces.py
```

This generates `known_faces.pkl`, which stores facial embeddings and associated identities.

### Step 2: Start Real-Time Recognition

```bash
python realtime_face_recog.py
```

The webcam feed will open and recognized faces will be labeled in real time.

## Performance Optimizations

* OpenCV DNN used for efficient CPU-based face detection.
* Face tracking reduces redundant recognition calls.
* Facial embeddings are precomputed and stored for faster lookup.
* Lightweight architecture suitable for laptops without a dedicated GPU.

## Limitations

* Recognition accuracy depends on image quality and lighting conditions.
* Extreme face angles may not be detected reliably.
* Performance may decrease with very large face databases.

## Author

**Viraj Akili**
