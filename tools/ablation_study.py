"""Ablation study: measure each post-processing step's individual contribution."""
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pretty_midi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(__file__), '..')

# Load the pipa-tuned Basic Pitch output (starting point)
midi = pretty_midi.PrettyMIDI(os.path.join(BASE, 'outputs', 'basic_pitch', 'bp_pipa_tuned.mid'))
notes = list(midi.instruments[0].notes)

print(f'Starting point: {len(notes)} notes (pipa-tuned Basic Pitch)')
print()

# Define each processing step
def filter_pitch_range(notes, low=38, high=84):
    return [n for n in notes if low <= n.pitch <= high]

def filter_short_notes(notes, min_dur=0.06):
    return [n for n in notes if (n.end - n.start) >= min_dur]

def merge_close_notes(notes, max_gap=0.05):
    if not notes:
        return notes
    notes = sorted(notes, key=lambda n: (n.pitch, n.start))
    merged = [notes[0]]
    for note in notes[1:]:
        prev = merged[-1]
        if note.pitch == prev.pitch and note.start - prev.end <= max_gap:
            prev.end = max(prev.end, note.end)
            prev.velocity = max(prev.velocity, note.velocity)
        else:
            merged.append(note)
    return merged

def merge_tremolos(notes, max_gap=0.10):
    if not notes:
        return notes
    notes = sorted(notes, key=lambda n: (n.pitch, n.start))
    merged = [notes[0]]
    for note in notes[1:]:
        prev = merged[-1]
        if note.pitch == prev.pitch and note.start - prev.end <= max_gap:
            prev.end = max(prev.end, note.end)
        else:
            merged.append(note)
    return merged

def smooth_pitches(notes, window=5, threshold=3):
    if len(notes) < window:
        return notes
    pitches = np.array([n.pitch for n in notes])
    from scipy.signal import medfilt
    smoothed = medfilt(pitches, kernel_size=window)
    result = []
    for i, note in enumerate(notes):
        if abs(pitches[i] - smoothed[i]) <= threshold:
            result.append(note)
    return result

# Run ablation: apply steps one at a time, measuring cumulative effect
steps = [
    ('1. Raw pipa-tuned', lambda ns: ns),
    ('2. + Pitch range filter (MIDI 38-84)', lambda ns: filter_pitch_range(ns)),
    ('3. + Short note filter (>60ms)', lambda ns: filter_short_notes(filter_pitch_range(ns))),
    ('4. + Gap filling (<50ms)', lambda ns: merge_close_notes(filter_short_notes(filter_pitch_range(ns)))),
    ('5. + Tremolo merging (<100ms)', lambda ns: merge_tremolos(merge_close_notes(filter_short_notes(filter_pitch_range(ns))))),
    ('6. + Pitch smoothing (medfilt)', lambda ns: smooth_pitches(merge_tremolos(merge_close_notes(filter_short_notes(filter_pitch_range(ns)))))),
]

# Also run ablation: remove one step at a time from full pipeline
print('=' * 70)
print('CUMULATIVE ABLATION (adding steps one by one)')
print('=' * 70)
print(f'{"Step":<42} {"Notes":>6} {"Removed":>8} {"Mean Dur":>10} {"Pitch Range":>15}')
print('-' * 70)

cumulative_data = []
current_notes = list(notes)

for step_name, step_fn in steps:
    current_notes = step_fn(list(notes))
    n = len(current_notes)
    removed = len(notes) - n
    durations = [nn.end - nn.start for nn in current_notes] if current_notes else [0]
    pitches = [nn.pitch for nn in current_notes] if current_notes else [0]
    mean_dur = np.mean(durations)
    p_range = f'{min(pitches)}-{max(pitches)}' if current_notes else 'N/A'
    cumulative_data.append({
        'step': step_name, 'notes': n, 'removed': removed,
        'mean_duration': mean_dur, 'pitch_range': p_range,
        'durations': durations, 'pitches': pitches,
    })
    print(f'{step_name:<42} {n:>6} {removed:>+8} {mean_dur:>9.3f}s {p_range:>15}')

# Leave-one-out ablation
print()
print('=' * 70)
print('LEAVE-ONE-OUT ABLATION (remove one step from full pipeline)')
print('=' * 70)
print(f'{"Configuration":<45} {"Notes":>6} {"Mean Dur":>10}')
print('-' * 70)

def full_pipeline(ns, skip_step=None):
    ns = list(ns)
    if skip_step != 'pitch_range':
        ns = filter_pitch_range(ns)
    if skip_step != 'short_notes':
        ns = filter_short_notes(ns)
    if skip_step != 'gap_filling':
        ns = merge_close_notes(ns)
    if skip_step != 'tremolo':
        ns = merge_tremolos(ns)
    if skip_step != 'smoothing':
        ns = smooth_pitches(ns)
    return ns

loo_configs = [
    ('Full pipeline (all steps)', None),
    ('Remove pitch range filter', 'pitch_range'),
    ('Remove short note filter', 'short_notes'),
    ('Remove gap filling', 'gap_filling'),
    ('Remove tremolo merging', 'tremolo'),
    ('Remove pitch smoothing', 'smoothing'),
    ('No pipeline (raw pipa-tuned)', 'all'),
]

loo_data = []
for config_name, skip in loo_configs:
    if skip == 'all':
        result = list(notes)
    else:
        result = full_pipeline(list(notes), skip)
    n = len(result)
    durations = [nn.end - nn.start for nn in result] if result else [0]
    mean_dur = np.mean(durations)
    loo_data.append({'config': config_name, 'notes': n, 'mean_duration': mean_dur})
    print(f'{config_name:<45} {n:>6} {mean_dur:>9.3f}s')

# --- Visualization ---
PLOT_DIR = os.path.join(BASE, 'evaluation', 'plots')

# Cumulative ablation bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

labels = [d['step'].split('. ', 1)[1] for d in cumulative_data]
note_counts = [d['notes'] for d in cumulative_data]
colors = ['#95a5a6', '#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#2ecc71']

bars = ax1.bar(range(len(labels)), note_counts, color=colors, edgecolor='white', linewidth=1.5)
ax1.set_xticks(range(len(labels)))
ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
ax1.set_ylabel('Note Count', fontsize=12)
ax1.set_title('Cumulative Pipeline Effect', fontsize=13, fontweight='bold')
for bar, count in zip(bars, note_counts):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
             str(count), ha='center', va='bottom', fontweight='bold', fontsize=11)

mean_durs = [d['mean_duration'] * 1000 for d in cumulative_data]  # ms
ax2.plot(range(len(labels)), mean_durs, 'o-', color='#2c3e50', linewidth=2, markersize=8)
ax2.set_xticks(range(len(labels)))
ax2.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
ax2.set_ylabel('Mean Duration (ms)', fontsize=12)
ax2.set_title('Mean Note Duration per Step', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'ablation_cumulative.png'), dpi=150, bbox_inches='tight')
plt.close()
print('\nSaved ablation_cumulative.png')

# Leave-one-out bar chart
fig, ax = plt.subplots(figsize=(12, 6))
loo_labels = [d['config'] for d in loo_data]
loo_notes = [d['notes'] for d in loo_data]
loo_colors = ['#2ecc71'] + ['#e74c3c'] * 5 + ['#95a5a6']

bars = ax.barh(range(len(loo_labels)), loo_notes, color=loo_colors, edgecolor='white', linewidth=1.5)
ax.set_yticks(range(len(loo_labels)))
ax.set_yticklabels(loo_labels, fontsize=10)
ax.set_xlabel('Note Count', fontsize=12)
ax.set_title('Leave-One-Out Ablation: Impact of Each Pipeline Step', fontsize=13, fontweight='bold')
for bar, count in zip(bars, loo_notes):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            str(count), ha='left', va='center', fontweight='bold')
ax.invert_yaxis()
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'ablation_leave_one_out.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved ablation_leave_one_out.png')

print('\nDone!')
