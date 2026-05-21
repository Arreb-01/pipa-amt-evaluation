"""Run librosa pyin as a baseline pitch tracker for comparison with AMT tools."""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import librosa
import numpy as np
import pretty_midi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

AUDIO = os.path.join(os.path.dirname(__file__), '..', '火把节之夜', '火把节之夜_mono.wav')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'pyin_baseline')
PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'evaluation', 'plots')

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

print("=" * 60)
print("Running librosa pyin baseline on pipa excerpt")
print("=" * 60)

y, sr = librosa.load(AUDIO, sr=44100)
duration = len(y) / sr
print(f"Audio: {duration:.2f}s, sr={sr}")

# Run pyin with pipa-appropriate frequency range
# Pipa range is roughly D2-A5, so use C2-C7
print("Running pyin pitch estimation...")
f0, voiced_flags, voiced_probs = librosa.pyin(
    y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
)

# Get time array for frames
times = librosa.times_like(f0, sr=sr)
print(f"Total frames: {len(f0)}, Voiced frames: {np.sum(~np.isnan(f0))}")

# Convert f0 to MIDI notes
voiced_mask = ~np.isnan(f0)
midi_pitches = np.zeros_like(f0)
midi_pitches[voiced_mask] = librosa.hz_to_midi(f0[voiced_mask])

# Quantize to semitones
midi_rounded = np.round(midi_pitches).astype(int)

# Create MIDI file from pitch track
# Group consecutive frames with same pitch into notes
midi_obj = pretty_midi.PrettyMIDI()
instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

hop_length = 512  # default for pyin
frame_duration = hop_length / sr  # ~0.023s per frame

# Detect note onsets/offsets
notes = []
current_pitch = None
note_start = None

for i in range(len(midi_rounded)):
    if voiced_mask[i]:
        pitch = midi_rounded[i]
        if current_pitch is None:
            # Note onset
            current_pitch = pitch
            note_start = times[i]
        elif abs(pitch - current_pitch) > 0.5:
            # Pitch changed - end current note, start new one
            note_end = times[i]
            if note_end - note_start >= 0.05:  # minimum 50ms
                notes.append(pretty_midi.Note(
                    velocity=64, pitch=int(current_pitch),
                    start=note_start, end=note_end
                ))
            current_pitch = pitch
            note_start = times[i]
    else:
        # Unvoiced - end current note
        if current_pitch is not None:
            note_end = times[i]
            if note_end - note_start >= 0.05:
                notes.append(pretty_midi.Note(
                    velocity=64, pitch=int(current_pitch),
                    start=note_start, end=note_end
                ))
            current_pitch = None
            note_start = None

# Handle last note
if current_pitch is not None:
    note_end = times[-1]
    if note_end - note_start >= 0.05:
        notes.append(pretty_midi.Note(
            velocity=64, pitch=int(current_pitch),
            start=note_start, end=note_end
        ))

instrument.notes = notes
midi_obj.instruments.append(instrument)

# Save MIDI
midi_path = os.path.join(OUT_DIR, 'pyin_baseline_output.mid')
midi_obj.write(midi_path)
print(f"Saved MIDI to: {midi_path}")

# Analysis
print(f"\n--- pyin Baseline Output Analysis ---")
print(f"Number of notes: {len(notes)}")
if notes:
    pitches = [n.pitch for n in notes]
    durations = [n.end - n.start for n in notes]
    print(f"Pitch range: {min(pitches)} - {max(pitches)} (MIDI)")
    print(f"Duration range: {min(durations):.3f}s - {max(durations):.3f}s")
    print(f"Mean duration: {np.mean(durations):.3f}s")

# Generate piano roll
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[1, 1])

# f0 contour (continuous)
ax1.plot(times[voiced_mask], midi_pitches[voiced_mask], '.', markersize=1, alpha=0.5, color='steelblue')
ax1.set_xlim(0, duration)
ax1.set_ylim(20, 100)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('MIDI Pitch (continuous)')
ax1.set_title('librosa pyin - Continuous Pitch Contour', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, pos: pretty_midi.note_number_to_name(int(x)) if 21 <= int(x) <= 108 else ''))

# Quantized piano roll
for note in notes:
    rect = plt.Rectangle((note.start, note.pitch - 0.5), note.end - note.start, 1,
                          linewidth=0.5, edgecolor='red', facecolor='red', alpha=0.7)
    ax2.add_patch(rect)
ax2.set_xlim(0, duration)
ax2.set_ylim(20, 100)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('MIDI Pitch')
ax2.set_title('librosa pyin - Quantized Note Output', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, pos: pretty_midi.note_number_to_name(int(x)) if 21 <= int(x) <= 108 else ''))

plt.tight_layout()
plot_path = os.path.join(PLOT_DIR, 'pyin_baseline_pianoroll.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved piano roll to: {plot_path}")

# Save note CSV
csv_path = os.path.join(OUT_DIR, 'pyin_baseline_notes.csv')
with open(csv_path, 'w') as f:
    f.write('start,end,pitch,velocity,duration\n')
    for note in notes:
        f.write(f'{note.start:.4f},{note.end:.4f},{note.pitch},{note.velocity},{note.end-note.start:.4f}\n')
print(f"Saved note CSV to: {csv_path}")

print("\nDone!")
