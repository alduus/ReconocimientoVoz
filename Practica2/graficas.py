import librosa
import numpy as np
import pandas as pd
import os

def extract_features_segmentado(audio_path, segment_ms=100):
    y, sr = librosa.load(audio_path, sr=None)
    segment_size = int(sr * (segment_ms / 1000.0))
    total_segments = len(y) // segment_size

    features_list = []

    for i in range(total_segments):
        start = i * segment_size
        end = start + segment_size
        segment = y[start:end]

        if len(segment) < segment_size:
            break  # descartar segmento incompleto final

        timestamp_sec = i * (segment_ms / 1000.0)

        # ===================== FEATURES REFINADAS =====================
        rms = np.mean(librosa.feature.rms(y=segment))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=segment))
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=segment, sr=sr, roll_percent=0.85))

        # MFCCs
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfcc, axis=1)

        # Energy entropy (opcional)
        frame_energy = segment**2
        frame_energy /= np.sum(frame_energy) + 1e-6
        entropy = -np.sum(frame_energy * np.log2(frame_energy + 1e-6))

        feature_dict = {
            'timestamp_sec': timestamp_sec,
            'rms': rms,
            'zcr': zcr,
            'spectral_centroid': spec_centroid,
            'spectral_bandwidth': spec_bw,
            'spectral_rolloff': rolloff,
            'energy_entropy': entropy,
        }

        for j, mfcc_val in enumerate(mfcc_means):
            feature_dict[f'mfcc_{j+1}'] = mfcc_val

        features_list.append(feature_dict)

    return pd.DataFrame(features_list)

# 👉 Cargar todos los audios y concatenar
audio_files = [
    "casahogar.wav",
    "escuela.wav",
    "lacasaesgrande.wav",
    "miperrosesaliodecasa.wav",
    "parque.wav"
]

all_segments = []

for audio_path in audio_files:
    features_df = extract_features_segmentado(audio_path)
    features_df['file'] = os.path.basename(audio_path)
    all_segments.append(features_df)

df_all = pd.concat(all_segments, ignore_index=True)

# 👉 Guardar CSV con features refinadas
df_all.to_csv("audio_segment_features_refined.csv", index=False)
print("✅ Features refinadas guardadas en audio_segment_features_refined.csv")
