import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from skimage.feature import graycomatrix, graycoprops
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import filedialog


# ektraksi GLCM
def glcm_feature_extraction(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256))
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    return contrast, homogeneity, energy


# baca dataset
df = pd.read_csv('data_lamun_baru.csv')
X = df[['contrast', 'homogeneity', 'energy']]
y = df['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# nilai K 
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


# baca gambar
def klasifikasi_gambar(image_path, img_label, result_text):
    img = cv2.imread(image_path)
    if img is None:
        result_text.set("Gagal membaca gambar.")
        return

    # preview gambar
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize((200, 200))
    img_tk = ImageTk.PhotoImage(img_pil)
    img_label.configure(image=img_tk)
    img_label.image = img_tk

    # ekstraksi fitur dan prediksi
    contrast, homogeneity, energy = glcm_feature_extraction(img)
    fitur_df = pd.DataFrame([[contrast, homogeneity, energy]], columns=['contrast', 'homogeneity', 'energy'])
    features_scaled = scaler.transform(fitur_df)

    jenis_knn = knn.predict(features_scaled)[0]

    hasil = f"""Data Ekstraksi:
Contrast    : {contrast:.2f}
Homogeneity : {homogeneity:.2f}
Energy      : {energy:.2f}

Jenis Lamun: {jenis_knn}
"""
    result_text.set(hasil)
    

# GUI
def tampilkan_gui():
    root = tk.Tk()
    root.title("Klasifikasi Jenis Lamun")
    root.geometry("500x500")

    tk.Label(root, text="Klasifikasi Jenis Lamun", font=("Helvetica", 12)).pack(pady=10)

    img_label = tk.Label(root)
    img_label.pack()

    result_text = tk.StringVar()
    result_label = tk.Label(root, textvariable=result_text, justify="left", font=("Courier", 10))
    result_label.pack(pady=10)

    def on_browse():
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            klasifikasi_gambar(file_path, img_label, result_text)
    # input gambar
    browse_btn = tk.Button(root, text="Buka Gambar", command=on_browse)
    browse_btn.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    tampilkan_gui()
