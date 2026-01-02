import tensorflow as tf
import numpy as np
import os

# 1. VERİ SETİNİ İNDİR VE HAZIRLA (MNIST)
print("MNIST veri seti indiriliyor...")
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize et (0-1 arası) ve Boyutlandır (28x28x1)
train_images = train_images.reshape((60000, 28, 28, 1)) / 255.0
test_images = test_images.reshape((10000, 28, 28, 1)) / 255.0

# 2. MODELİ OLUŞTUR (Basit CNN - Section 12.9)
print("Model oluşturuluyor...")
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. EĞİT (Hızlı olsun diye 2 epoch yeter)
print("Model eğitiliyor (Biraz sürebilir)...")
model.fit(train_images, train_labels, epochs=2, validation_split=0.1)

# 4. TFLITE DÖNÜŞTÜRME (Quantization - STM32 için)
print("STM32 uyumlu .tflite dosyası oluşturuluyor...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Modeli Kaydet
with open('mnist_model.tflite', 'wb') as f:
    f.write(tflite_model)

# 5. OFFLINE VERİ SETİ OLUŞTURMA (C Header Dosyası - KRİTİK KISIM)
# Test setinden rastgele bir '7' rakamını bulup C dizisine çeviriyoruz.
print("Test verisinden örnek bir rakam (7) seçilip C koduna çevriliyor...")
sample_idx = 0
while test_labels[sample_idx] != 7: # Bir tane 7 bulana kadar ara
    sample_idx += 1

sample_image = test_images[sample_idx].flatten()

header_content = f"""
/* EE4065 - Q2 Offline Data Set (Section 12.9) */
/* Auto-generated from MNIST Test Set */
/* Label: {test_labels[sample_idx]} (Digit 7) */

#ifndef DIGIT_DATA_H
#define DIGIT_DATA_H

#include <stdint.h>

/* 28x28 = 784 piksellik görüntü verisi */
const float offline_digit_image[784] = {{
"""

for i, val in enumerate(sample_image):
    if i % 12 == 0: header_content += "\n    "
    header_content += f"{val:.4f}f, "

header_content += "\n};\n\n#endif // DIGIT_DATA_H"

with open('digit_data.h', 'w') as f:
    f.write(header_content)

print("\n" + "="*50)
print("BAŞARILI! 🎉")
print("1. 'mnist_model.tflite' oluştu (GitHub'a atılacak)")
print("2. 'digit_data.h' oluştu (STM32 projesine atılacak)")
print("="*50)