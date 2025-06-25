import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle


def train_and_save_svm_model(embedding_file, model_output_path):
    # Load face embeddings and labels
    data = np.load(embedding_file)
    EMBEDDED_X = data['arr_0']
    Y = data['arr_1']

    if EMBEDDED_X.size == 0 or Y.size == 0:
        print("Embedding data or labels are empty.")
        return

    # Encode labels
    encoder = LabelEncoder()
    Y = encoder.fit_transform(Y)

    # Check number of unique classes
    unique_classes = np.unique(Y)
    if len(unique_classes) < 2:
        print("Not enough variety in the dataset. At least two different classes are required.")
        return

    # Split data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y, shuffle=True, random_state=17)

    # Train the SVM model
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, Y_train)

    # Save the model
    with open(model_output_path, 'wb') as f:
        pickle.dump(model, f)
    print("Model has been saved successfully.")


# Call the function to train and save the model
train_and_save_svm_model('resources/FaceReco/faces_embeddings_done_4classes.npz',
                         'resources/FaceReco/svm_model_160x160.pkl')
