import os
import cv2
import pickle
import tkinter as tk
from tkinter import messagebox

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()



def get_button(window, text, color, command, fg='white'):
    button = tk.Button(
        window,
        text=text,
        activebackground="black",
        activeforeground="white",
        fg=fg,
        bg=color,
        command=command,
        height=2,
        width=20,
        font=('Helvetica bold', 20)
    )
    return button


def get_img_label(window):
    label = tk.Label(window)
    label.grid(row=0, column=0)
    return label


def get_text_label(window, text):
    label = tk.Label(window, text=text)
    label.config(font=("sans-serif", 21), justify="left")
    return label


def get_entry_text(window):
    inputtxt = tk.Text(window, height=2, width=15, font=("Arial", 32))
    return inputtxt


def msg_box(title, description):
    messagebox.showinfo(title, description)


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    return faces


def recognize(img, recognizer, labels):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_face(img)

    if len(faces) == 0:
        return 'no_persons_found'

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        label, confidence = recognizer.predict(face)
        if confidence < 50:  # Confidence threshold
            return labels[label]
    return 'unknown_person'


def save_user(name, img, recognizer, db_dir, labels):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_face(img)

    if len(faces) == 0:
        return False

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        label = len(labels)
        recognizer.update([face], [label])
        labels.append(name)
        with open(os.path.join(db_dir, 'labels.pickle'), 'wb') as f:
            pickle.dump(labels, f)
        recognizer.write(os.path.join(db_dir, 'recognizer.yml'))
    return True
