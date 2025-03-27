# src/traditional_pipeline.py
import os
import cv2
import numpy as np
from imutils import paths
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Set dataset paths
dataset_path = r"E:\Projects\QR Code Authentication Model\dataset"
first_print_path = os.path.join(dataset_path, "First_Print")
second_print_path = os.path.join(dataset_path, "Second_Print")

def load_dataset():
    imagePaths = []
    labels = []
    for imgPath in paths.list_images(first_print_path):
        imagePaths.append(imgPath)
        labels.append(0)  # label 0 for First Print (original)
    for imgPath in paths.list_images(second_print_path):
        imagePaths.append(imgPath)
        labels.append(1)  # label 1 for Second Print (counterfeit)
    return imagePaths, labels

def extract_features(imagePath):
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), block_norm='L2-Hys',
                      visualize=True)
    return features

def main():
    print("[INFO] Loading dataset...")
    imagePaths, labels = load_dataset()
    data = []
    for path in imagePaths:
        features = extract_features(path)
        data.append(features)
    data = np.array(data)
    labels = np.array(labels)
    
    # Split the dataset into training and testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)
    
    # Train the SVM classifier
    print("[INFO] Training SVM classifier...")
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(trainX, trainY)
    
    # Evaluate the model
    preds = model.predict(testX)
    print("Classification Report:\n", classification_report(testY, preds))
    print("Confusion Matrix:\n", confusion_matrix(testY, preds))
    
    # Save the trained model
    joblib.dump(model, "svm_qr_classifier.joblib")
    print("[INFO] Model saved as svm_qr_classifier.joblib")

if __name__ == "__main__":
    main()
