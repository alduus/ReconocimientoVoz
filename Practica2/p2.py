import librosa
import librosa.feature
import numpy as np
import pandas as pd
import os

# Parámetros
FRAME_SIZE = 0.1  # 100 ms
MFCC_COUNT = 13

# Lista de archivos de audio (ajusta las rutas si los tienes localmente)
audio_files = [
    "parque.wav",
    "miperrosesaliodecasa.wav",
    "lacasaesgrande.wav",
    "escuela.wav",
    "casahogar.wav"
]

# Función para extraer características
def extract_features(file_path, frame_size=FRAME_SIZE):
    y, sr = librosa.load(file_path, sr=None)
    frame_length = int(sr * frame_size)
    hop_length = frame_length  # sin solapamiento
    
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_COUNT, hop_length=hop_length, n_fft=frame_length)

    # Alineamos todo
    n_frames = rms.shape[0]
    data = []
    for i in range(n_frames):
        timestamp = i * frame_size
        row = {
            'file': os.path.basename(file_path),
            'timestamp_sec': round(timestamp, 2),
            'rms': rms[i],
            'zcr': zcr[i]
        }
        for j in range(MFCC_COUNT):
            row[f'mfcc_{j+1}'] = mfcc[j][i]
        data.append(row)
    
    return pd.DataFrame(data)

# Procesar todos los audios
all_data = []
for audio in audio_files:
    print(f"Procesando: {audio}")
    df = extract_features(audio)
    all_data.append(df)

# Concatenar todo
features_df = pd.concat(all_data, ignore_index=True)

# Guardar como CSV
features_df.to_csv("audio_segment_features.csv", index=False)
print("✔ Características extraídas y guardadas en 'audio_segment_features.csv'")
