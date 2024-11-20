# Tugas-Pengolahan-Citra-Digital

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow  # Untuk menampilkan gambar di Colab

def detect_ripeness(image_path):
    # Baca gambar
    image = cv2.imread(image_path)
    if image is None:
        print("Gambar tidak ditemukan!")
        return
    
    # Konversi ke ruang warna HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Rentang warna untuk buah matang (merah dan kuning)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Rentang warna untuk buah tidak matang (hijau)
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    # Masking warna matang
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_ripe = cv2.bitwise_or(mask_red, mask_yellow)

    # Masking warna tidak matang
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Hitung area warna
    ripe_area = cv2.countNonZero(mask_ripe)
    unripe_area = cv2.countNonZero(mask_green)

    # Total area
    total_area = ripe_area + unripe_area
    if total_area == 0:
        print("Tidak ada area warna yang terdeteksi!")
        return

    # Persentase warna matang dan tidak matang
    ripe_percentage = (ripe_area / total_area) * 100
    unripe_percentage = (unripe_area / total_area) * 100

    # Output hasil
    print(f"Buah matang: {ripe_percentage:.2f}%")
    print(f"Buah tidak matang: {unripe_percentage:.2f}%")

    # Tampilkan hasil masking
    plt.figure(figsize=(10, 10))

    # Gambar asli
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    # Mask warna matang
    plt.subplot(1, 3, 2)
    plt.title("Ripe Mask")
    plt.imshow(mask_ripe, cmap="gray")
    plt.axis("off")

    # Mask warna tidak matang
    plt.subplot(1, 3, 3)
    plt.title("Unripe Mask")
    plt.imshow(mask_green, cmap="gray")
    plt.axis("off")

    plt.show()

# Upload gambar ke Colab
from google.colab import files

uploaded = files.upload()  # Upload file gambar
image_path = list(uploaded.keys())[0]  # Ambil nama file yang diupload

# Jalankan deteksi
detect_ripeness(image_path)
