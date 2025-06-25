import cv2
import dlib
import numpy as np
import os
import pickle
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from ultralytics import YOLO

# dlib yüz algılayıcıyı yükle
face_model = YOLO("yolo/yolov8n-face.pt")
detector = dlib.get_frontal_face_detector()
facenet = FaceNet()
faces_embeddings = np.load("resources/FaceReco/faces_embeddings_done_4classes.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
model = pickle.load(open("resources/FaceReco/svm_model_160x160.pkl", 'rb'))

# Proje klasöründe bir dizin oluştur
output_dir = "detected_faces"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def detect_and_recognize(frame):
    # dlib ile yüzleri tespit et
    results = face_model(frame)
    faces = results[0].boxes
    if faces:
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = map(int, face.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Yüz bölgesini kırp
            face_img = frame[y1:y2, x1:x2]
            if face_img.size > 0:
                face_img_resized = cv2.resize(face_img, (160, 160))

                # FaceNet ile yüz embedding'lerini al
                face_img_expanded = np.expand_dims(face_img_resized, axis=0)
                embeddings = facenet.embeddings(face_img_expanded)
                probs = model.predict_proba(embeddings)[0]
                max_prob = np.max(probs)
                face_name = model.predict(embeddings)

                if max_prob < 0.5:
                    final_name = "Unknown"
                else:
                    final_name = encoder.inverse_transform(face_name)[0]
                print(final_name)
    return frame



def main():
    # Kameradan görüntü al
    cap = cv2.VideoCapture(0)  # 0 varsayılan kamerayı temsil eder

    if not cap.isOpened():
        print("Kamera açılamıyor!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Görüntü alınamıyor!")
            break

        frame_with_detections = detect_and_recognize(frame)

        cv2.imshow("Face Detection and Recognition", frame_with_detections)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
