# Real-Time Face Recognition System

## Overview
A CPU-optimized real-time face recognition system using OpenCV DNN for face detection and embedding-based recognition using face_recognition.

## Features
- Real-time webcam face detection
- Face recognition using embeddings
- Duplicate prevention via tracking
- CPU-optimized pipeline
- Clean modular structure

## Tech Stack
- Python
- OpenCV (DNN)
- face_recognition
- NumPy

## How It Works
1. Face detection using OpenCV DNN (ResNet SSD).
2. Extract facial embeddings.
3. Compare with known face encodings.
4. Assign identity.
5. Track faces to prevent duplicates.

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
Step 1: Encode known faces
```bash
python encode_known_faces.py
```
Step 2: Run real-time recognition
```bash
python realtime_face_recog.py
```
## Author
Viraj Akili
