from ultralytics import YOLO
import cv2 as cv
import numpy as np
import threading
from keras_facenet import FaceNet
import pickle
from sklearn.preprocessing import LabelEncoder
import os
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
from collections import defaultdict
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("CUDA version:", tf.sysconfig.get_build_info()["cuda_version"])
print("cuDNN version:", tf.sysconfig.get_build_info()["cudnn_version"])
print("Loading YOLOv9 model...")
yolo_model = YOLO("yolo/yolov10n.pt")
print("YOLOv9 model loaded.")

print("Loading YOLOv8 face model...")
face_model = YOLO("yolo/yolov8n-face.pt")
print("YOLOv8 face model loaded.")

print("Loading FaceNet model and face recognition model...")
facenet = FaceNet()
faces_embeddings = np.load("resources/FaceReco/faces_embeddings_done_4classes.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
recognition_model = pickle.load(open("resources/FaceReco/svm_model_160x160.pkl", 'rb'))
print("FaceNet and face recognition models loaded.")

print("Loading anti-spoofing model...")
antispoofing_model = AntiSpoofPredict(device_id=0)
antispoof_model_dir = "./resources/anti_spoof_models"
print("Anti-spoofing model loaded.")

# Global frame
frame = None

# Global dictionary to store recognized faces
recognized_faces = defaultdict(lambda: {"name": "Unknown", "confidence": 0, "anti_spoofing": (0, 0)})

def capture_frames(cap):
    global frame
    while cap.isOpened():
        ret, new_frame = cap.read()
        if not ret:
            print("Camera connection lost or frame not captured. Reconnecting...")
            cap.release()
            cap = cv.VideoCapture(0)
            continue
        frame = new_frame

def predict_antispoofing(frame, bbox):
    image_cropper = CropImage()
    prediction = np.zeros((1, 3))
    for model_name in os.listdir(antispoof_model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": frame,
            "bbox": bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        prediction += antispoofing_model.predict(img, os.path.join(antispoof_model_dir, model_name))
    label = np.argmax(prediction)
    return label, prediction[0][label] / 2

def recognize_face(face_img_resized):
    face_img_expanded = np.expand_dims(face_img_resized, axis=0)
    embeddings = facenet.embeddings(face_img_expanded)
    probs = recognition_model.predict_proba(embeddings)[0]
    max_prob = np.max(probs)
    face_name = recognition_model.predict(embeddings)

    if max_prob < 0.5:
        final_name = "Unknown"
    else:
        final_name = encoder.inverse_transform(face_name)[0]
    return final_name, max_prob

def detect_objects(frame):
    annotated_frame = frame.copy()
    h, w = frame.shape[:2]
    results = yolo_model(frame)
    print("Objects detected.")

    person_boxes = []
    for result in results[0].boxes:
        if result.cls[0] == 0:  # Class ID 0 is usually 'person' in COCO dataset
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            person_boxes.append((x1, y1, x2, y2))

    person_labels = [f"Person-{i + 1}" for i in range(len(person_boxes))]

    def process_face(face_img, face_img_resized, person_label):
        face_id = f"{person_label}-{left}-{top}-{right}-{bottom}"
        results = {}

        def antispoofing_thread():
            results['antispoofing'] = predict_antispoofing(face_img, [left, top, right-left, bottom-top])

        def recognition_thread():
            results['recognition'] = recognize_face(face_img_resized)

        thread_antispoofing = threading.Thread(target=antispoofing_thread)
        thread_recognition = threading.Thread(target=recognition_thread)

        thread_antispoofing.start()
        thread_recognition.start()

        thread_antispoofing.join()
        thread_recognition.join()

        antispoofing_result = results.get('antispoofing', (0, 0))
        final_name, max_prob = results.get('recognition', ("Unknown", 0))

        if antispoofing_result[0] == 1 and antispoofing_result[1] > 0.80:
            recognized_faces[face_id] = {"name": final_name, "confidence": max_prob, "anti_spoofing": antispoofing_result}

        return antispoofing_result, final_name, max_prob

    for (x1, y1, x2, y2), person_label in zip(person_boxes, person_labels):
        cv.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        person_roi = frame[y1:y2, x1:x2]
        face_results = face_model(person_roi)
        print("Face detection results obtained.")

        face_boxes = []
        for face_result in face_results[0].boxes:
            if face_result.cls[0] == 0:  # Face class usually ID 0 in YOLO face model
                fx1, fy1, fx2, fy2 = map(int, face_result.xyxy[0])
                face_boxes.append((x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2))

        for (left, top, right, bottom) in face_boxes:
            face_img = frame[top:bottom, left:right]
            face_img_resized = cv.resize(face_img, (160, 160))

            face_id = f"{person_label}-{left}-{top}-{right}-{bottom}"
            if face_id in recognized_faces:
                face_label = f"{recognized_faces[face_id]['name']} - {recognized_faces[face_id]['confidence']:.2f}"
                cv.rectangle(annotated_frame, (left, top), (right, bottom), (255, 0, 0), 2)
                cv.putText(annotated_frame, face_label, (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            else:
                antispoofing_result, face_name, max_prob = process_face(person_roi, face_img_resized, person_label)

                if antispoofing_result[0] == 1 and antispoofing_result[1] > 0.962:
                    face_label = f"{face_name} - {max_prob:.2f}"
                    cv.rectangle(annotated_frame, (left, top), (right, bottom), (255, 0, 0), 2)
                    cv.putText(annotated_frame, face_label, (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    print("AntiSpoofing result:", antispoofing_result)
                    print("Real face detected:", face_label)
                else:
                    face_label = "Spoof Detected"
                    cv.rectangle(annotated_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv.putText(annotated_frame, face_label, (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    print("AntiSpoofing result:", antispoofing_result)
                    print("Spoof:", face_label)

    cv.imshow("Real-time Object Detection and Face Recognition", annotated_frame)

def main():
    global frame
    cap = cv.VideoCapture(1)
    if not cap.isOpened():
        print("Failed to connect to camera. Please check the connection.")
        return

    frame_thread = threading.Thread(target=capture_frames, args=(cap,))
    frame_thread.start()

    while True:
        if frame is not None:
            detect_objects(frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
