import cv2 as cv
import numpy as np
import os
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import shutil

class FACELOADING:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160, 160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()

    def extract_face(self, filename):
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x, y, w, h = self.detector.detect_faces(img)[0]['box']
        x, y = abs(x), abs(y)
        face = img[y:y + h, x:x + w]
        face_arr = cv.resize(face, self.target_size)
        return face_arr

    def load_faces(self, dir):
        FACES = []
        filenames = []
        for im_name in os.listdir(dir):
            try:
                path = os.path.join(dir, im_name)
                single_face = self.extract_face(path)
                FACES.append(single_face)
                filenames.append(path)
            except Exception as e:
                print(f"Error processing {im_name}: {e}")
                pass
        return FACES, filenames

    def load_classes(self):
        filenames_dict = {}
        for sub_dir in os.listdir(self.directory):
            path = os.path.join(self.directory, sub_dir)
            FACES, filenames = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            print(f"Successfully loaded: {len(labels)} images for class '{sub_dir}'")
            self.X.extend(FACES)
            self.Y.extend(labels)
            filenames_dict[sub_dir] = filenames

        return np.asarray(self.X), np.asarray(self.Y), filenames_dict

    def plot_images(self):
        plt.figure(figsize=(18, 16))
        for num, image in enumerate(self.X):
            ncols = 3
            nrows = len(self.Y) // ncols + 1
            plt.subplot(nrows, ncols, num + 1)
            plt.imshow(image)
            plt.axis('off')

# Load faces and compute embeddings
faceloading = FACELOADING("dataset/new_users")
X, Y, filenames_dict = faceloading.load_classes()

embedder = FaceNet()

# Try to load existing embeddings
try:
    existing_data = np.load('resources/FaceReco/faces_embeddings_done_4classes.npz')
    EMBEDDED_X_existing, Y_existing = existing_data['arr_0'], existing_data['arr_1']
    EMBEDDED_X = list(EMBEDDED_X_existing)
except FileNotFoundError:
    EMBEDDED_X = []
    Y_existing = []

def get_embedding(face_img):
    face_img = face_img.astype('float32')  # 3D (160x160x3)
    face_img = np.expand_dims(face_img, axis=0)  # 4D (1, 160, 160, 3)
    yhat = embedder.embeddings(face_img)
    return yhat[0]  # 512D embedding

# Compute embeddings for new faces
new_embeddings = []
for img in X:
    embedding = get_embedding(img)
    new_embeddings.append(embedding)

# Append new embeddings to existing ones
EMBEDDED_X.extend(new_embeddings)

# Update labels
if len(Y_existing) > 0:
    Y_existing = np.concatenate((Y_existing, Y), axis=0)
else:
    Y_existing = Y

# Save updated embeddings and labels
np.savez_compressed('resources/FaceReco/faces_embeddings_done_4classes.npz', EMBEDDED_X, Y_existing)

# Move newly processed images to dataset/users
for label, filenames in filenames_dict.items():
    user_dir = os.path.join("dataset/users", label)
    os.makedirs(user_dir, exist_ok=True)
    for filename in filenames:
        shutil.move(filename, user_dir)

# Optional: Plot loaded face images
faceloading.plot_images()
plt.show()
