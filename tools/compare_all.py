"""Generate comprehensive comparison plots: spectrogram + all tool outputs."""
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

AUDIO = os.path.join(os.path.dirname(__file__), '..', '火把节之夜', '火把节之夜_mono.wav')
PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'evaluation', 'plots')

# Load outputs
bp_midi = pretty_midi.PrettyMIDI(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'basic_pitch', 'basic_pitch_output.mid'))
pyin_midi = pretty_midi.PrettyMIDI(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'pyin_baseline', 'pyin_baseline_output.mid'))

# Load audio
y, sr = librosa.load(AUDIO, sr=44100)
duration = len(y) / sr

# Color scheme
colors = {'basic_pitch': '#E74C3C', 'pyin': '#3498DB'}

fig, axes = plt.subplots(4, 1, figsize=(20, 18), height_ratios=[1.2, 1, 1, 1])

# 1. Spectrogram
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axes[0])
axes[0].set_title('Original Audio - Spectrogram', fontsize=14, fontweight='bold')

# 2. Chromagram
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', ax=axes[1])
axes[1].set_title('Chromagram (CQT) - Pitch Class Distribution', fontsize=14, fontweight='bold')

# Helper function to plot piano roll
def plot_pianoroll(ax, midi_data, title, color, alpha=0.7):
    for inst in midi_data.instruments:
        for note in inst.notes:
            rect = plt.Rectangle((note.start, note.pitch - 0.5), note.end - note.start, 1,
                                  linewidth=0.3, edgecolor=color, facecolor=color, alpha=alpha)
            ax.add_patch(rect)
    ax.set_xlim(0, duration)
    ax.set_ylim(30, 95)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('MIDI Pitch')
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: pretty_midi.note_number_to_name(int(x)) if 21 <= int(x) <= 108 else ''))

# 3. Basic Pitch
bp_notes = bp_midi.instruments[0].notes if bp_midi.instruments else []
plot_pianoroll(axes[2], bp_midi,
               f'Basic Pitch ({len(bp_notes)} notes, range: {min(n.pitch for n in bp_notes)}-{max(n.pitch for n in bp_notes)})',
               colors['basic_pitch'])

# 4. pyin baseline
pyin_notes = pyin_midi.instruments[0].notes if pyin_midi.instruments else []
plot_pianoroll(axes[3], pyin_midi,
               f'librosa pyin ({len(pyin_notes)} notes, range: {min(n.pitch for n in pyin_notes)}-{max(n.pitch for n in pyin_notes)})',
               colors['pyin'])

axes[3].set_xlabel('Time (s)', fontsize=12)

plt.tight_layout()
plot_path = os.path.join(PLOT_DIR, 'all_tools_comparison.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved comparison plot to: {plot_path}")

# Also create an overlay comparison (both tools on same axes)
fig, ax = plt.subplots(figsize=(20, 8))
plot_pianoroll(ax, bp_midi, '', colors['basic_pitch'], alpha=0.5)
for inst in pyin_midi.instruments:
    for note in inst.notes:
        rect = plt.Rectangle((note.start, note.pitch - 0.5), note.end - note.start, 1,
                              linewidth=0.3, edgecolor=colors['pyin'], facecolor=colors['pyin'], alpha=0.3)
        ax.add_patch(rect)
ax.set_title('Overlay: Basic Pitch (red) vs librosa pyin (blue)', fontsize=14, fontweight='bold')
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_xlim(0, duration)
ax.set_ylim(30, 95)
ax.grid(True, alpha=0.2)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors['basic_pitch'], alpha=0.5, label=f'Basic Pitch ({len(bp_notes)} notes)'),
                   Patch(facecolor=colors['pyin'], alpha=0.5, label=f'pyin ({len(pyin_notes)} notes)')]
ax.legend(handles=legend_elements, fontsize=12)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, pos: pretty_midi.note_number_to_name(int(x)) if 21 <= int(x) <= 108 else ''))

overlay_path = os.path.join(PLOT_DIR, 'tools_overlay.png')
plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved overlay plot to: {overlay_path}")

# Note density comparison per second
fig, ax = plt.subplots(figsize=(16, 5))
window = 1.0  # 1 second windows
times_bins = np.arange(0, duration, window)

bp_density = []
pyin_density = []
for t in times_bins:
    bp_count = sum(1 for n in bp_notes if t <= n.start < t + window)
    pyin_count = sum(1 for n in pyin_notes if t <= n.start < t + window)
    bp_density.append(bp_count)
    pyin_density.append(pyin_count)

ax.bar(times_bins - 0.15, bp_density, width=0.3, color=colors['basic_pitch'], alpha=0.7, label='Basic Pitch')
ax.bar(times_bins + 0.15, pyin_density, width=0.3, color=colors['pyin'], alpha=0.7, label='pyin')
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Notes per second', fontsize=12)
ax.set_title('Note Density Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.2)

density_path = os.path.join(PLOT_DIR, 'note_density.png')
plt.savefig(density_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved density plot to: {density_path}")

print("\nDone!")
