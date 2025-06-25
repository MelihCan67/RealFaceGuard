import cv2 as cv
import numpy as np
import os
import pickle
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder


def recognize_face(frame, threshold=0.4):
    facenet = FaceNet()
    faces_embeddings = np.load("resources/FaceReco/faces_embeddings_done_4classes.npz")
    Y = faces_embeddings['arr_1']
    encoder = LabelEncoder()
    encoder.fit(Y)
    haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    model = pickle.load(open("resources/FaceReco/svm_model_160x160.pkl", 'rb'))

    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)

    for x, y, w, h in faces:
        img = rgb_img[y:y + h, x:x + w]
        img = cv.resize(img, (160, 160))  # 1x160x160x3
        img = np.expand_dims(img, axis=0)
        ypred = facenet.embeddings(img)

        # Tahmin olasılıklarını hesaplayın
        probs = model.predict_proba(ypred)[0]
        max_prob = np.max(probs)
        face_name = model.predict(ypred)

        if max_prob < threshold:
            final_name = "Unknown"
        else:
            final_name = encoder.inverse_transform(face_name)[0]

        return final_name

    return "Unknown"
