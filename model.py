import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


df_train = pd.read_csv("training.csv")

print("Kolom di data:", df_train.columns)

X_train = df_train.drop(columns="label")  
y_train = df_train["label"]

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_train)

acc = accuracy_score(y_train, y_pred)
print(f"Akurasi model KNN: {acc * 100:.2f}%")
print("\nLaporan Klasifikasi:")
print(classification_report(y_train, y_pred))