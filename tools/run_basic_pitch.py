"""Run Basic Pitch on the pipa excerpt and save results."""
import os
import sys

# Force UTF-8 output
os.environ['PYTHONIOENCODING'] = 'utf-8'

from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import pretty_midi
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AUDIO = os.path.join(os.path.dirname(__file__), '..', '火把节之夜', '火把节之夜_mono.wav')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'basic_pitch')
PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'evaluation', 'plots')

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

print("=" * 60)
print("Running Basic Pitch on pipa excerpt")
print("=" * 60)

# Run Basic Pitch inference
model_path = ICASSP_2022_MODEL_PATH
print(f"Model: {model_path}")
print(f"Audio: {AUDIO}")

model_output, midi_data, note_events = predict(AUDIO)

# Save MIDI
midi_path = os.path.join(OUT_DIR, 'basic_pitch_output.mid')
midi_data.write(midi_path)
print(f"\nSaved MIDI to: {midi_path}")

# Analyze output
print(f"\n--- Basic Pitch Output Analysis ---")
print(f"Number of instruments: {len(midi_data.instruments)}")
for i, inst in enumerate(midi_data.instruments):
    notes = inst.notes
    print(f"Instrument {i}: {len(notes)} notes")
    if notes:
        pitches = [n.pitch for n in notes]
        durations = [n.end - n.start for n in notes]
        velocities = [n.velocity for n in notes]
        print(f"  Pitch range: {min(pitches)} - {max(pitches)} (MIDI)")
        print(f"  Duration range: {min(durations):.3f}s - {max(durations):.3f}s")
        print(f"  Mean velocity: {np.mean(velocities):.1f}")
        print(f"  Mean duration: {np.mean(durations):.3f}s")

# Generate piano roll plot
fig, ax = plt.subplots(figsize=(16, 6))
for inst in midi_data.instruments:
    for note in inst.notes:
        rect = plt.Rectangle((note.start, note.pitch - 0.5), note.end - note.start, 1,
                              linewidth=0.5, edgecolor='steelblue', facecolor='steelblue', alpha=0.7)
        ax.add_patch(rect)

ax.set_xlim(0, midi_data.get_end_time())
ax.set_ylim(20, 100)
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('MIDI Pitch', fontsize=12)
ax.set_title('Basic Pitch - Piano Roll Output (Pipa: Night of the Torch Festival)', fontsize=14)
ax.grid(True, alpha=0.3)

# Add pitch labels on y-axis
import matplotlib.ticker as ticker
ax.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, pos: pretty_midi.note_number_to_name(int(x)) if 21 <= int(x) <= 108 else ''))

plt.tight_layout()
plot_path = os.path.join(PLOT_DIR, 'basic_pitch_pianoroll.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved piano roll to: {plot_path}")

# Also create a comparison: spectrogram + piano roll overlay
y, sr = librosa.load(AUDIO, sr=44100)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[1, 1])

# Spectrogram
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax1)
ax1.set_title('Original Audio - Spectrogram', fontsize=14)

# Piano roll
for inst in midi_data.instruments:
    for note in inst.notes:
        rect = plt.Rectangle((note.start, note.pitch - 0.5), note.end - note.start, 1,
                              linewidth=0.5, edgecolor='red', facecolor='red', alpha=0.7)
        ax2.add_patch(rect)
ax2.set_xlim(0, midi_data.get_end_time())
ax2.set_ylim(20, 100)
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('MIDI Pitch', fontsize=12)
ax2.set_title('Basic Pitch - Transcription', fontsize=14)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, pos: pretty_midi.note_number_to_name(int(x)) if 21 <= int(x) <= 108 else ''))
ax2.grid(True, alpha=0.3)

plt.tight_layout()
compare_path = os.path.join(PLOT_DIR, 'basic_pitch_comparison.png')
plt.savefig(compare_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved comparison to: {compare_path}")

# Save note data as CSV for further analysis
csv_path = os.path.join(OUT_DIR, 'basic_pitch_notes.csv')
with open(csv_path, 'w') as f:
    f.write('start,end,pitch,velocity,duration\n')
    for inst in midi_data.instruments:
        for note in inst.notes:
            f.write(f'{note.start:.4f},{note.end:.4f},{note.pitch},{note.velocity},{note.end-note.start:.4f}\n')
print(f"Saved note CSV to: {csv_path}")

print("\nDone!")
