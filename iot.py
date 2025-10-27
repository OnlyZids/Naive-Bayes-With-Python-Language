import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# Dataset IoT Inkubator Itik
data = {
    "Suhu":[37.5,37.8,37.2,36.9,35.8,36.0,35.5,36.2],
    "Kelembapan":[60,58,65,67,70,68,72,66],
    "Durasi":[28,27,29,30,32,33,31,30],
    "Frekuensi":[4,4,3,3,2,2,2,3],
    "Hasil":["Sukses","Sukses","Sukses","Sukses","Gagal","Gagal","Gagal","Gagal"]
}
df = pd.DataFrame(data)

X = df[["Suhu","Kelembapan","Durasi","Frekuensi"]]
y = df["Hasil"]

le = LabelEncoder()
y_enc = le.fit_transform(y)

model = GaussianNB()
model.fit(X, y_enc)

# Data uji
sample = pd.DataFrame([[37.0, 64, 29, 3]], columns=["Suhu","Kelembapan","Durasi","Frekuensi"])
pred = model.predict(sample)
prob = model.predict_proba(sample)

print("=== Prediksi Data IoT ===")
print(f"Data uji : {list(sample.iloc[0])}")
print(f"Hasil Prediksi : {le.inverse_transform(pred)[0]}")
print(f"Probabilitas [Gagal, Sukses] : {prob[0]}")
print(f"Probabilitas Sukses : {prob[0][1]*100:.6f}%")
print(f"Probabilitas Gagal  : {prob[0][0]*100:.6f}%")
