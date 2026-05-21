"""Final comprehensive comparison of all three AMT tools + improved version."""
import os
import warnings
warnings.filterwarnings('ignore')

import librosa
import numpy as np
import pretty_midi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

AUDIO = os.path.join(os.path.dirname(__file__), '..', 'tools', 'audio_mono.wav')
PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'evaluation', 'plots')

# Load all MIDI outputs
bp = pretty_midi.PrettyMIDI(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'basic_pitch', 'basic_pitch_output.mid'))
pyin = pretty_midi.PrettyMIDI(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'pyin_baseline', 'pyin_baseline_output.mid'))
pt = pretty_midi.PrettyMIDI(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'piano_transcription', 'piano_transcription_output.mid'))
improved = pretty_midi.PrettyMIDI(os.path.join(os.path.dirname(__file__), '..', 'improvement', 'improved_basic_pitch.mid'))

y, sr = librosa.load(AUDIO, sr=44100)
duration = 94.3

tools = [
    ('Basic Pitch', bp, '#E74C3C'),
    ('librosa pyin (Baseline)', pyin, '#3498DB'),
    ('Piano Transcription (ByteDance)', pt, '#9B59B6'),
    ('Improved Basic Pitch (Post-processed)', improved, '#2ECC71'),
]

# Full 5-panel comparison
fig, axes = plt.subplots(5, 1, figsize=(22, 22), height_ratios=[1.2, 1, 1, 1, 1])

# 1. Spectrogram
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axes[0])
axes[0].set_title('Original Audio — Spectrogram (Night of the Torch Festival, Pipa)', fontsize=14, fontweight='bold')

# 2-5. Tool outputs
for idx, (name, midi_data, color) in enumerate(tools):
    ax = axes[idx + 1]
    notes = midi_data.instruments[0].notes if midi_data.instruments else []
    for note in notes:
        rect = plt.Rectangle((note.start, note.pitch - 0.5), note.end - note.start, 1,
                              linewidth=0.3, edgecolor=color, facecolor=color, alpha=0.7)
        ax.add_patch(rect)

    pitch_range = f'{min(n.pitch for n in notes)}-{max(n.pitch for n in notes)}' if notes else 'N/A'
    mean_dur = f'{np.mean([n.end-n.start for n in notes]):.3f}s' if notes else 'N/A'
    ax.set_title(f'{name} — {len(notes)} notes, pitch range: {pitch_range}, mean duration: {mean_dur}',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, duration)
    ax.set_ylim(30, 95)
    ax.set_ylabel('MIDI')
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: pretty_midi.note_number_to_name(int(x)) if 21 <= int(x) <= 108 else ''))

axes[-1].set_xlabel('Time (s)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'final_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved final_comparison.png')

# Note density comparison
fig, ax = plt.subplots(figsize=(18, 6))
window = 1.0
times_bins = np.arange(0, duration, window)

for name, midi_data, color in tools:
    notes = midi_data.instruments[0].notes if midi_data.instruments else []
    density = [sum(1 for n in notes if t <= n.start < t + window) for t in times_bins]
    ax.plot(times_bins, density, color=color, linewidth=1.5, label=name, alpha=0.8)

ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Notes per second', fontsize=12)
ax.set_title('Note Onset Density Over Time', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'final_density_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved final_density_comparison.png')

# Pitch histogram comparison
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
for idx, (name, midi_data, color) in enumerate(tools):
    notes = midi_data.instruments[0].notes if midi_data.instruments else []
    pitches = [n.pitch for n in notes]
    if pitches:
        axes[idx].hist(pitches, bins=range(30, 96), color=color, alpha=0.7, edgecolor='white')
    axes[idx].set_title(name, fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('MIDI Pitch')
    axes[idx].xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: pretty_midi.note_number_to_name(int(x)) if 21 <= int(x) <= 108 else ''))
axes[0].set_ylabel('Note Count')
plt.suptitle('Pitch Distribution Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'pitch_histogram.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved pitch_histogram.png')

print('Done!')
