import numpy as np
import tensorflow as tf

# 1. Eğittiğimiz Modeli Yükle (STM32'nin yapacağı işi yapıyoruz)
interpreter = tf.lite.Interpreter(model_path="kws_model_pruned.tflite")
interpreter.allocate_tensors()

# Giriş ve Çıkış detaylarını al
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. Rastgele Bir Ses Verisi Üret (Mikrofon verisi simülasyonu)
# Giriş şekli: (1, 49, 10, 1) -> 49 zaman adımı, 10 frekans
input_shape = input_details[0]['shape']
fake_audio_data = np.random.random_sample(input_shape).astype(np.float32)

# Veriyi modele ver
interpreter.set_tensor(input_details[0]['index'], fake_audio_data)

# 3. Modeli Çalıştır (Inference)
interpreter.invoke()

# 4. Sonucu Al
output_data = interpreter.get_tensor(output_details[0]['index'])[0]

# 5. Sonucu Ekrana Yazdır (İşte bu senin RESULT'ın!)
classes = ["Silence", "Unknown", "YES", "NO"]
predicted_index = np.argmax(output_data)
confidence = output_data[predicted_index]

print("\n" + "="*40)
print("   Q1 KEYWORD SPOTTING - TEST RESULT   ")
print("="*40)
print(f"Input Data Shape: {input_shape}")
print("-" * 40)
print("Model Predictions:")
for i, prob in enumerate(output_data):
    print(f" > {classes[i]}: %{prob*100:.2f}")

print("-" * 40)
print(f"FINAL DECISION: '{classes[predicted_index]}' detected!")
print(f"CONFIDENCE:     %{confidence*100:.2f}")
print("="*40 + "\n")
