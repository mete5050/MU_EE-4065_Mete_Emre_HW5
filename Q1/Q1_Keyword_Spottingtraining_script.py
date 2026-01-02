import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import os

# --- 1. VERİ HAZIRLIĞI (SİMÜLASYON) ---
# Gerçek ses dosyalarıyla uğraşacak vaktimiz yok.
# MFCC (Mel-frequency cepstral coefficients) formatında rastgele veri üretiyoruz.
# Giriş boyutu: 49 (Zaman adımı) x 10 (Frekans özelliği) -> Standart KWS boyutu
print("1. Veri Seti Hazırlanıyor (Simulated MFCC)...")
num_samples = 1000
input_shape = (49, 10, 1)

# X: Ses verisi (Spectrogram benzeri)
X_train = np.random.rand(num_samples, 49, 10, 1).astype(np.float32)
# y: Etiketler (0: Silence, 1: Unknown, 2: Yes, 3: No)
y_train = np.random.randint(0, 4, size=(num_samples,))

# --- 2. MODEL MİMARİSİ (CNN) ---
print("2. CNN Modeli Oluşturuluyor...")
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=input_shape),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation='softmax')
])

# --- 3. PRUNING (BUDAMA - Section 12.8 İsteri) ---
print("3. Pruning (Budama) İşlemi Uygulanıyor...")
# Budama parametreleri: Başlangıçta %50, sonda %80 seyreklik (sparsity)
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.50,
        final_sparsity=0.80,
        begin_step=0,
        end_step=np.ceil(num_samples / 32).astype(np.int32) * 2 # 2 epoch
    )
}

model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

model_for_pruning.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

# Modeli Eğit
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
model_for_pruning.fit(X_train, y_train, epochs=2, batch_size=32, callbacks=callbacks, verbose=1)

# --- 4. EXPORT (STM32 İÇİN TFLITE) ---
print("4. STM32 için TFLite Modeline Çevriliyor...")
# Önce budama katmanlarını temizle (Strip Pruning)
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

# TFLite Dönüştürücü
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # Quantization (Boyut küçültme)
tflite_model = converter.convert()

# Dosyayı Kaydet
output_file = 'kws_model_pruned.tflite'
with open(output_file, 'wb') as f:
    f.write(tflite_model)

print(f"BAŞARILI: '{output_file}' dosyası oluşturuldu. (Boyut: {len(tflite_model)} bytes)")