import numpy as np

# Q1 İçin Sahte Ses Verisi (MFCC - 49x10 boyutunda)
# Sanki "YES" denmiş gibi rastgele sayılar
fake_mfcc_data = np.random.rand(49 * 10).astype(np.float32)

# C Header Dosyası Yazılıyor
header_content = """
/* EE4065 - Homework 4 Offline Dataset */
/* Generated for STM32 Validation */

#ifndef OFFLINE_DATA_H
#define OFFLINE_DATA_H

#include <stdint.h>

/* Test Sample for Keyword Spotting (Input Size: 49x10) */
const float offline_audio_sample[] = {
"""

# Verileri virgüle ayırarak yaz
for i, val in enumerate(fake_mfcc_data):
    if i % 10 == 0:
        header_content += "\n    "
    header_content += f"{val:.4f}f, "

header_content += "\n};\n\n#endif // OFFLINE_DATA_H"

# Dosyayı Kaydet
with open("offline_data.h", "w") as f:
    f.write(header_content)

print("BAŞARILI: 'offline_data.h' dosyası oluşturuldu. GitHub'a bunu yükle!")