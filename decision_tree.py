# ---- Bagian Klasifikasi ----
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans

# Baca dataset
data = pd.read_csv("data.csv")

# Pisahkan fitur dan label
X = data[["Cuaca", "Suhu", "Kelembaban", "Berangin"]]
y = data["Main"]

# Ubah ke numerik
X_encoded = X.apply(lambda col: pd.factorize(col)[0])
y_encoded = pd.factorize(y)[0]

# Split data jadi 80% latih dan 20% uji
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)


# Pakai data fitur yang sudah di-encode
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X_encoded)

# Buat model Decision Tree
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi
print("\n=== HASIL KLASIFIKASI ===")
print("Akurasi :", accuracy_score(y_test, y_pred))
print("Laporan:\n", classification_report(y_test, y_pred))

# Tambahkan hasil cluster ke dataset
data["Cluster"] = kmeans.labels_
print("\n=== HASIL CLUSTERING (K-Means) ===")
print(data[["Cuaca", "Suhu", "Kelembaban", "Berangin", "Cluster"]])
