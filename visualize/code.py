import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Membaca data dari file Excel
file_path = "data.xlsx"  # Ganti dengan path file Excel
sheet_name = "Sheet1"  # Sesuaikan dengan nama sheet jika perlu
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Mengambil data x, y, dan error lateral
x_ref = data["x"].values
y_ref = data["y"].values
error_lateral = data["error_lateral"].values

# Batas kesalahan lateral maksimal (garis merah kanan)
lateral_max = 20  # Misalkan batas lateral maksimum adalah 20 satuan
x_right = x_ref
y_right = y_ref + lateral_max

# Garis hitam (trajektori aktual dengan error lateral)
x_actual = x_ref
y_actual = y_ref + error_lateral

# Plot hasil visualisasi
plt.figure(figsize=(8, 6))
plt.plot(x_ref, y_ref, 'r', label='Garis Merah Kiri (Referensi)')
plt.plot(x_right, y_right, 'r', linestyle='dashed', label='Garis Merah Kanan (Batas Maks)')
plt.plot(x_actual, y_actual, 'k', label='Garis Hitam (Trajektori Aktual)')

plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.title('Visualisasi Trajektori')
plt.legend()
plt.grid()
plt.show()
