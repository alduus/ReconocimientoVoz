import pandas as pd

# Cargar el CSV generado en el paso anterior
df = pd.read_csv("audio_segment_features.csv")

# ---------- Umbrales heurísticos (ajústalos según distribución real de tus datos) ----------
RMS_SILENCE = 0.01       # por debajo de esto se considera silencio
RMS_VOICE = 0.03         # por encima de esto se considera voz
ZCR_VOICE = 0.05         # valor típico ZCR para voz humana
ZCR_SILENCE = 0.01       # por debajo = silencio claro

# ---------- Función de etiquetado ----------
def label_segment(row):
    if row['rms'] < RMS_SILENCE and row['zcr'] < ZCR_SILENCE:
        return 'S'  # Silencio
    elif row['rms'] >= RMS_VOICE and row['zcr'] >= ZCR_VOICE:
        return 'V'  # Voz
    else:
        return 'U'  # No sonoro

# Aplicar etiquetas
df['label'] = df.apply(label_segment, axis=1)

# Guardar con etiquetas
df.to_csv("audio_segment_features_labeled.csv", index=False)
print("✔ Etiquetado heurístico inicial completo. Guardado en 'audio_segment_features_labeled.csv'")
