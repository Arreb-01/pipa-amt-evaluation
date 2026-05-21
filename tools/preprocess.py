"""Phase 0: Audio preprocessing — convert stereo to mono, analyze audio properties."""
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import os

SRC = os.path.join(os.path.dirname(__file__), '..', '火把节之夜', '火把节之夜.wav')
DST = os.path.join(os.path.dirname(__file__), '..', '火把节之夜', '火把节之夜_mono.wav')
PLOT = os.path.join(os.path.dirname(__file__), '..', 'evaluation', 'plots', 'audio_overview.png')

# Load audio (librosa auto-converts to mono if mono=True)
y, sr = librosa.load(SRC, sr=44100, mono=True)
duration = len(y) / sr
print(f"Sample rate: {sr} Hz")
print(f"Duration: {duration:.2f}s ({duration/60:.2f} min)")
print(f"Samples: {len(y)}")
print(f"Peak amplitude: {np.max(np.abs(y)):.4f}")
print(f"RMS energy: {np.sqrt(np.mean(y**2)):.4f}")

# Save mono version
sf.write(DST, y, sr)
print(f"\nSaved mono audio to: {DST}")

# Generate overview plot
fig, axes = plt.subplots(3, 1, figsize=(16, 10))

# Waveform
t = np.arange(len(y)) / sr
axes[0].plot(t, y, linewidth=0.1, color='steelblue')
axes[0].set_title('Waveform', fontsize=14)
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')

# Spectrogram
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axes[1])
axes[1].set_title('Spectrogram', fontsize=14)

# Chromagram
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', ax=axes[2])
axes[2].set_title('Chromagram (CQT)', fontsize=14)

plt.tight_layout()
plt.savefig(PLOT, dpi=150, bbox_inches='tight')
print(f"Saved overview plot to: {PLOT}")

# Detect pitch range using pyin
print("\nEstimating pitch range with pyin...")
f0, voiced_flags, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
voiced_f0 = f0[~np.isnan(f0)]
if len(voiced_f0) > 0:
    low_note = librosa.hz_to_note(np.min(voiced_f0)).encode('ascii', 'replace').decode()
    high_note = librosa.hz_to_note(np.max(voiced_f0)).encode('ascii', 'replace').decode()
    med_note = librosa.hz_to_note(np.median(voiced_f0)).encode('ascii', 'replace').decode()
    print(f"Pitch range: {low_note} - {high_note}")
    print(f"  ({np.min(voiced_f0):.1f} Hz - {np.max(voiced_f0):.1f} Hz)")
    print(f"Median pitch: {med_note}")
