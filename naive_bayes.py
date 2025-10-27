# naive_bayes_kualitas.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# --- 1. Dataset (bisa diganti dengan file CSV/XLSX kamu) ---
# df = pd.read_csv("kualitas_air.csv")
data = {
    "pH": [7.5, 7.2, 7.8, 6.9, 6.2, 6.0, 5.8, 6.4],
    "DO": [6.8, 6.5, 7.0, 5.9, 4.1, 3.8, 3.5, 4.3],
    "Suhu": [25, 26, 24, 27, 29, 30, 31, 28],
    "Label": ["Baik", "Baik", "Baik", "Baik", "Buruk", "Buruk", "Buruk", "Buruk"]
}
df = pd.DataFrame(data)

# --- 2. Preprocessing ---
X = df[["pH", "DO", "Suhu"]].values
y = df["Label"].values

le = LabelEncoder()
y_enc = le.fit_transform(y)  # Baik=1, Buruk=0
print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# --- 3. Split data train-test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.25, stratify=y_enc, random_state=42
)

# --- 4. Train model Naive Bayes ---
model = GaussianNB()
model.fit(X_train, y_train)

# --- 5. Evaluasi model ---
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # Probabilitas kelas "Baik"

print("\n=== Evaluasi Model ===")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

if len(np.unique(y_test)) == 2:
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC Score: {auc:.4f}")

# --- 6. Cross Validation (4-fold) ---
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
scores = cross_val_score(model, X, y_enc, cv=cv, scoring="accuracy")

print("\n=== Cross Validation ===")
print("Fold accuracies:", np.round(scores, 4))
print("Mean CV accuracy:", np.mean(scores))

# --- 7. Prediksi untuk data baru ---
sample = pd.DataFrame([[7.0, 6.0, 26]], columns=["pH", "DO", "Suhu"])
pred = model.predict(sample)
prob = model.predict_proba(sample)

print("\n=== Prediksi Data Baru ===")
print(f"Data uji : pH={sample.iloc[0,0]}, DO={sample.iloc[0,1]}, Suhu={sample.iloc[0,2]}")
print(f"Hasil Prediksi : {le.inverse_transform(pred)[0]}")
print(f"Probabilitas [Baik, Buruk] : {prob[0]}")
print(f"Probabilitas Baik  : {prob[0][0]*100:.10f}%")
print(f"Probabilitas Buruk : {prob[0][1]*100:.10f}%")
print("Selesai.")
