# =====================================================
# REGRESI LINEAR SEDERHANA
# Pengaruh Durasi Media Sosial terhadap Produktivitas Belajar Mahasiswa
# =====================================================

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# =====================================================
# 1. Membaca file CSV
# =====================================================

df = pd.read_csv("dataset_media_sosial.csv", sep=';')

# =====================================================
# 2. Menentukan Kolom
# =====================================================

durasi_col = df.columns[0]
produktivitas_col = df.columns[1]

# =====================================================
# 3. Mengubah Durasi Menjadi Angka
# =====================================================

durasi_mapping = {
    "<1 jam": 1,
    "1-3 jam": 2,
    "4-6 jam": 3,
    "6 jam lebih": 4
}

df["Durasi"] = (
    df[durasi_col]
    .astype(str)
    .str.strip()
    .map(durasi_mapping)
)

# =====================================================
# 4. Mengubah Produktivitas Menjadi Angka
# =====================================================

df["Produktivitas"] = pd.to_numeric(
    df[produktivitas_col],
    errors="coerce"
)

# =====================================================
# 5. Menghapus Data Kosong
# =====================================================

data = df[["Durasi", "Produktivitas"]].dropna()

# =====================================================
# 6. Variabel X dan Y
# =====================================================

X = data[["Durasi"]]
Y = data["Produktivitas"]

X_reg = sm.add_constant(X)

# =====================================================
# 7. Membuat Model Regresi
# =====================================================

model = sm.OLS(Y, X_reg).fit()

# =====================================================
# 8. Menampilkan Hasil Regresi
# =====================================================

print("===== HASIL REGRESI =====")
print(model.summary())

# =====================================================
# 9. Persamaan Regresi
# =====================================================

a = model.params.iloc[0]
b = model.params.iloc[1]

print("\nPersamaan Regresi:")
print(f"Y = {a:.3f} + ({b:.3f})X")

# =====================================================
# 10. Koefisien Determinasi
# =====================================================

print("\nR Square :", round(model.rsquared, 3))
print("Adjusted R Square :", round(model.rsquared_adj, 3))

# =====================================================
# 11. Uji t
# =====================================================

print("\nUji t")
print("t hitung :", round(model.tvalues.iloc[1], 3))
print("p-value :", round(model.pvalues.iloc[1], 5))

# =====================================================
# 12. Uji F
# =====================================================

print("\nUji F")
print("F hitung :", round(model.fvalue, 3))
print("p-value :", round(model.f_pvalue, 5))

# =====================================================
# 13. Menyimpan Hasil Analisis ke TXT
# =====================================================

with open("hasil_analisis.txt", "w", encoding="utf-8") as f:

    f.write("=== HASIL ANALISIS REGRESI LINEAR SEDERHANA ===\n\n")

    f.write("Nama Kolom Dataset:\n")
    for col in df.columns:
        f.write(f"- {col}\n")

    f.write(f"\nJumlah Data : {len(data)}\n\n")

    f.write("===== HASIL REGRESI =====\n")
    f.write(str(model.summary()))
    f.write("\n\n")

    f.write("Persamaan Regresi\n")
    f.write(f"Y = {a:.3f} + ({b:.3f})X\n\n")

    f.write(f"R Square : {model.rsquared:.3f}\n")
    f.write(f"Adjusted R Square : {model.rsquared_adj:.3f}\n\n")

    f.write("Uji t\n")
    f.write(f"t hitung : {model.tvalues.iloc[1]:.3f}\n")
    f.write(f"p-value : {model.pvalues.iloc[1]:.5f}\n\n")

    f.write("Uji F\n")
    f.write(f"F hitung : {model.fvalue:.3f}\n")
    f.write(f"p-value : {model.f_pvalue:.5f}\n")

# =====================================================
# 14. Prediksi
# =====================================================

Y_pred = model.predict(X_reg)

# =====================================================
# 15. Grafik Regresi
# =====================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    data["Durasi"],
    data["Produktivitas"],
    alpha=0.7,
    label="Data Responden"
)

data_sorted = data.sort_values("Durasi")

X_sorted = sm.add_constant(data_sorted[["Durasi"]])
Y_sorted_pred = model.predict(X_sorted)

plt.plot(
    data_sorted["Durasi"],
    Y_sorted_pred,
    linewidth=2,
    label="Garis Regresi"
)

plt.title("Pengaruh Durasi Media Sosial terhadap Produktivitas Belajar")
plt.xlabel("Durasi Penggunaan Media Sosial")
plt.ylabel("Produktivitas Belajar")

plt.xticks(
    [1, 2, 3, 4],
    ["<1 jam", "1-3 jam", "4-6 jam", "6 jam lebih"]
)

plt.legend()
plt.grid(True)

# Simpan grafik ke folder proyek
plt.savefig("grafik_regresi.png", dpi=300, bbox_inches="tight")

print("\nHasil analisis berhasil disimpan:")
print("- hasil_analisis.txt")
print("- grafik_regresi.png")

plt.show()