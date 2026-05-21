"""Audio pre-processing: denoising + normalization before AMT."""
import os
import warnings
warnings.filterwarnings('ignore')

import librosa
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AUDIO = os.path.join(os.path.dirname(__file__), 'audio_mono.wav')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'preprocessed')
PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'evaluation', 'plots')

os.makedirs(OUT_DIR, exist_ok=True)

y, sr = librosa.load(AUDIO, sr=44100)
print(f'Loaded: {len(y)/sr:.1f}s, sr={sr}')

# --- Pre-processing Pipeline ---
# 1. Peak normalization
y_norm = y / np.max(np.abs(y))

# 2. Spectral gating denoising using librosa
#    Estimate noise floor from first 0.5s (assuming initial silence/noise)
noise_sample = y_norm[:int(0.5 * sr)]
stft_noise = np.mean(np.abs(librosa.stft(noise_sample)), axis=1)

S = librosa.stft(y_norm)
S_mag = np.abs(S)
S_phase = np.angle(S)

# Apply spectral gate: suppress bins below noise threshold
threshold = stft_noise[:, np.newaxis] * 1.5
mask = S_mag > threshold
S_denoised = S_mag * mask

# Reconstruct signal
y_denoised = librosa.istft(S_denoised * np.exp(1j * S_phase))

# 3. Re-normalize after denoising
y_denoised = y_denoised / np.max(np.abs(y_denoised))

# Save preprocessed audio
out_path = os.path.join(OUT_DIR, 'audio_denoised.wav')
sf.write(out_path, y_denoised, sr)
print(f'Saved denoised audio: {out_path}')

# Also save to tools/ for Basic Pitch to use
bp_path = os.path.join(os.path.dirname(__file__), 'audio_denoised.wav')
sf.write(bp_path, y_denoised, sr)
print(f'Saved for Basic Pitch: {bp_path}')

# --- Visualization ---
fig, axes = plt.subplots(3, 1, figsize=(18, 10))

# Original spectrogram
D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D_orig, sr=sr, x_axis='time', y_axis='hz', ax=axes[0])
axes[0].set_title('Original Audio — Spectrogram', fontsize=12, fontweight='bold')

# Denoised spectrogram
D_denoised = librosa.amplitude_to_db(np.abs(librosa.stft(y_denoised)), ref=np.max)
librosa.display.specshow(D_denoised, sr=sr, x_axis='time', y_axis='hz', ax=axes[1])
axes[1].set_title('Denoised Audio — Spectrogram', fontsize=12, fontweight='bold')

# Difference
D_diff = D_orig - D_denoised
librosa.display.specshow(D_diff, sr=sr, x_axis='time', y_axis='hz', ax=axes[2])
axes[2].set_title('Removed Noise (Difference)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'preprocessing_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved preprocessing_comparison.png')

# Print stats
rms_orig = np.sqrt(np.mean(y**2))
rms_denoised = np.sqrt(np.mean(y_denoised**2))
print(f'RMS: original={rms_orig:.4f}, denoised={rms_denoised:.4f}')
print(f'Noise reduction: {(1 - rms_denoised/rms_orig)*100:.1f}%')
print('Done!')
