# encode_known_faces.py
import os
import face_recognition
import pickle

KNOWN_DIR = r"C:\Users\admin\OneDrive\Desktop\Image processing\known_faces"
OUTPUT_FILE = r"C:\Users\admin\OneDrive\Desktop\Image processing\known_faces.pkl"
TOLERANCE = 0.6   # used later; not for encoding step

known_encodings = []
known_names = []

# Walk each subfolder (person)
for person_name in os.listdir(KNOWN_DIR):
    person_dir = os.path.join(KNOWN_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue
    # for each image of that person
    for filename in os.listdir(person_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(person_dir, filename)
        image = face_recognition.load_image_file(img_path)
        # detect faces (list of face locations)
        boxes = face_recognition.face_locations(image, model="hog")  # or "cnn" if you have GPU/dlib cnn
        if len(boxes) == 0:
            print("No face found in", img_path)
            continue
        # compute encodings (one per detected face). We assume first face is the person.
        encs = face_recognition.face_encodings(image, boxes)
        if len(encs) > 0:
            known_encodings.append(encs[0])
            known_names.append(person_name)
            print("Encoded:", img_path, "->", person_name)

# Save to file
data = {"encodings": known_encodings, "names": known_names}
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(data, f)
print("Saved", len(known_encodings), "encodings to", OUTPUT_FILE)
