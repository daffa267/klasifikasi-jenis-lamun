import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tkinter import filedialog, Tk
from skimage.feature import graycomatrix, graycoprops


def extract_glcm_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    
    energy = graycoprops(glcm, 'energy')[0].mean()
    contrast = graycoprops(glcm, 'contrast')[0].mean()
    entropy = graycoprops(glcm, 'dissimilarity')[0].mean()
    idm = graycoprops(glcm, 'homogeneity')[0].mean()
    
    return [energy, contrast, entropy, idm]

def load_and_train_model():
    df_train = pd.read_csv("model.csv")
    
    X_train = df_train.drop(columns="label")
    y_train = df_train["label"]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_train.columns  

def predict_lamun_image():
    root = Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(title="Pilih Gambar", filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    
    if not file_path:
        print("Tidak ada gambar dipilih.")
        return
    
    image = cv2.imread(file_path)
    if image is None:
        print("Gagal membaca gambar.")
        return
    
    features = extract_glcm_features(image)
    
    model, scaler, feature_names = load_and_train_model()

    features_df = pd.DataFrame([features], columns=feature_names)
    features_scaled = scaler.transform(features_df)

    prediction = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)[0]

    print(f"\nPrediksi jenis lamun: {prediction[0]}")
    print("Probabilitas tiap kelas:")
    for idx, class_name in enumerate(model.classes_):
        print(f"- {class_name}: {probabilities[idx] * 100:.2f}%")

if __name__ == "__main__":
    predict_lamun_image()