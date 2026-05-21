"""Phase 3: Post-processing improvements on Basic Pitch output.

Improvements:
1. Short note filtering (remove likely false positives < min_duration)
2. Tremolo detection and merging (rapid repeated same-pitch notes)
3. Pitch smoothing (median filter to remove spurious pitch jumps)
4. Frequency range filtering (keep only pipa-appropriate range)
5. Gap filling (merge close notes of same pitch)
"""
import os
import warnings
warnings.filterwarnings('ignore')

import pretty_midi
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Paths
BP_MIDI = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'basic_pitch', 'bp_pipa_tuned.mid')
ORIG_MIDI = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'basic_pitch', 'basic_pitch_output.mid')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'improvement')
PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'evaluation', 'plots')

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Pipa-appropriate frequency range
PIPA_MIN_MIDI = 38   # ~D2
PIPA_MAX_MIDI = 84   # ~C6

def load_notes(midi_path):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    return midi_data.instruments[0].notes if midi_data.instruments else []

def notes_to_midi(notes, output_path):
    midi_obj = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes = notes
    midi_obj.instruments.append(inst)
    midi_obj.write(output_path)
    return midi_obj

def filter_short_notes(notes, min_duration=0.06):
    """Remove notes shorter than min_duration seconds."""
    return [n for n in notes if (n.end - n.start) >= min_duration]

def filter_pitch_range(notes, min_pitch=PIPA_MIN_MIDI, max_pitch=PIPA_MAX_MIDI):
    """Keep only notes within pipa range."""
    return [n for n in notes if min_pitch <= n.pitch <= max_pitch]

def merge_tremolos(notes, gap_threshold=0.08, same_pitch=True):
    """Merge rapid repeated notes (tremolo/轮指 detection).

    If two consecutive notes of the same pitch are separated by less than
    gap_threshold, merge them into a single sustained note.
    """
    if not notes:
        return notes

    # Sort by start time
    notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    merged = [notes[0]]

    for note in notes[1:]:
        prev = merged[-1]
        gap = note.start - prev.end

        if same_pitch and note.pitch == prev.pitch and gap < gap_threshold:
            # Merge: extend previous note
            prev.end = max(prev.end, note.end)
            prev.velocity = max(prev.velocity, note.velocity)
        elif note.pitch == prev.pitch and note.start - prev.start < gap_threshold:
            # Overlapping or very close same-pitch notes
            prev.end = max(prev.end, note.end)
        else:
            merged.append(note)

    return merged

def smooth_pitches(notes, window=3):
    """Apply median filtering to smooth pitch outliers.

    For notes that are isolated pitch outliers (jump up/down then back),
    replace with the median of their neighbors.
    """
    if len(notes) < window + 1:
        return notes

    notes = sorted(notes, key=lambda n: n.start)
    smoothed = list(notes)

    half_w = window // 2
    for i in range(half_w, len(notes) - half_w):
        neighborhood = [notes[j].pitch for j in range(i - half_w, i + half_w + 1)]
        median_pitch = int(np.median(neighborhood))
        # If note is an outlier (> 3 semitones from median), snap to median
        if abs(notes[i].pitch - median_pitch) > 3:
            smoothed[i] = pretty_midi.Note(
                velocity=notes[i].velocity,
                pitch=median_pitch,
                start=notes[i].start,
                end=notes[i].end
            )

    return smoothed

def merge_close_notes(notes, gap_threshold=0.05):
    """Merge notes of same pitch separated by very short gaps."""
    if not notes:
        return notes

    notes = sorted(notes, key=lambda n: (n.pitch, n.start))
    merged = [notes[0]]

    for note in notes[1:]:
        prev = merged[-1]
        if note.pitch == prev.pitch and (note.start - prev.end) < gap_threshold:
            prev.end = max(prev.end, note.end)
            prev.velocity = max(prev.velocity, note.velocity)
        else:
            merged.append(note)

    return merged

# Load original Basic Pitch output
print("=" * 60)
print("Phase 3: Post-processing Improvements")
print("=" * 60)

# Use pipa_tuned as base (already has better parameters)
base_notes = load_notes(BP_MIDI)
orig_notes = load_notes(ORIG_MIDI)
print(f"\nOriginal output: {len(orig_notes)} notes")
print(f"Base (pipa_tuned): {len(base_notes)} notes")

# Apply pipeline of improvements
print("\n--- Applying improvements ---")

# Step 1: Filter pitch range
notes = filter_pitch_range(base_notes)
print(f"After pitch range filter ({PIPA_MIN_MIDI}-{PIPA_MAX_MIDI}): {len(notes)} notes")

# Step 2: Filter short notes
notes = filter_short_notes(notes, min_duration=0.06)
print(f"After short note filter (>60ms): {len(notes)} notes")

# Step 3: Merge close same-pitch notes (gap filling)
notes = merge_close_notes(notes, gap_threshold=0.05)
print(f"After gap filling: {len(notes)} notes")

# Step 4: Tremolo merging
notes = merge_tremolos(notes, gap_threshold=0.10)
print(f"After tremolo merge: {len(notes)} notes")

# Step 5: Pitch smoothing
notes = smooth_pitches(notes, window=5)
print(f"After pitch smoothing: {len(notes)} notes")

# Sort by start time
notes = sorted(notes, key=lambda n: n.start)

# Save improved MIDI
improved_path = os.path.join(OUT_DIR, 'improved_basic_pitch.mid')
improved_midi = notes_to_midi(notes, improved_path)
print(f"\nSaved improved MIDI to: {improved_path}")

# Analysis
pitches = [n.pitch for n in notes]
durations = [n.end - n.start for n in notes]
print(f"\nImproved output: {len(notes)} notes")
print(f"Pitch range: {min(pitches)}-{max(pitches)} MIDI")
print(f"Duration range: {min(durations):.3f}-{max(durations):.3f}s (mean: {np.mean(durations):.3f}s)")

# Generate comparison plot: original vs improved
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 14))

# Original
for note in orig_notes:
    rect = plt.Rectangle((note.start, note.pitch - 0.5), note.end - note.start, 1,
                          linewidth=0.3, edgecolor='#E74C3C', facecolor='#E74C3C', alpha=0.7)
    ax1.add_patch(rect)
ax1.set_xlim(0, 95)
ax1.set_ylim(30, 95)
ax1.set_title(f'Original Basic Pitch ({len(orig_notes)} notes)', fontsize=13, fontweight='bold')
ax1.set_ylabel('MIDI Pitch')
ax1.grid(True, alpha=0.2)
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, pos: pretty_midi.note_number_to_name(int(x)) if 21 <= int(x) <= 108 else ''))

# Base (pipa_tuned)
for note in base_notes:
    rect = plt.Rectangle((note.start, note.pitch - 0.5), note.end - note.start, 1,
                          linewidth=0.3, edgecolor='#F39C12', facecolor='#F39C12', alpha=0.7)
    ax2.add_patch(rect)
ax2.set_xlim(0, 95)
ax2.set_ylim(30, 95)
ax2.set_title(f'Pipa-Tuned Basic Pitch ({len(base_notes)} notes)', fontsize=13, fontweight='bold')
ax2.set_ylabel('MIDI Pitch')
ax2.grid(True, alpha=0.2)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, pos: pretty_midi.note_number_to_name(int(x)) if 21 <= int(x) <= 108 else ''))

# Improved
for note in notes:
    rect = plt.Rectangle((note.start, note.pitch - 0.5), note.end - note.start, 1,
                          linewidth=0.3, edgecolor='#2ECC71', facecolor='#2ECC71', alpha=0.7)
    ax3.add_patch(rect)
ax3.set_xlim(0, 95)
ax3.set_ylim(30, 95)
ax3.set_title(f'Improved (Post-Processed) ({len(notes)} notes)', fontsize=13, fontweight='bold')
ax3.set_ylabel('MIDI Pitch')
ax3.set_xlabel('Time (s)')
ax3.grid(True, alpha=0.2)
ax3.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, pos: pretty_midi.note_number_to_name(int(x)) if 21 <= int(x) <= 108 else ''))

plt.tight_layout()
improve_plot = os.path.join(PLOT_DIR, 'improvement_comparison.png')
plt.savefig(improve_plot, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved improvement comparison to: {improve_plot}")

# Save improved notes CSV
csv_path = os.path.join(OUT_DIR, 'improved_notes.csv')
with open(csv_path, 'w') as f:
    f.write('start,end,pitch,velocity,duration\n')
    for note in notes:
        f.write(f'{note.start:.4f},{note.end:.4f},{note.pitch},{note.velocity},{note.end-note.start:.4f}\n')
print(f"Saved improved notes CSV to: {csv_path}")

# Summary statistics
print("\n--- Summary ---")
print(f"{'Metric':<25} {'Original':>12} {'Pipa-Tuned':>12} {'Improved':>12}")
print("-" * 65)
bp_notes = orig_notes
pt_notes = base_notes
for label, bn, pn, in_ in [
    ("Note count", len(bp_notes), len(pt_notes), len(notes)),
    ("Pitch min", min(n.pitch for n in bp_notes), min(n.pitch for n in pt_notes), min(pitches)),
    ("Pitch max", max(n.pitch for n in bp_notes), max(n.pitch for n in pt_notes), max(pitches)),
    ("Mean duration",
     np.mean([n.end-n.start for n in bp_notes]),
     np.mean([n.end-n.start for n in pt_notes]),
     np.mean(durations)),
]:
    print(f"{label:<25} {bn:>12} {pn:>12} {in_:>12.3f}" if isinstance(in_, float) else f"{label:<25} {bn:>12} {pn:>12} {in_:>12}")

print("\nDone!")
