"""mir_eval quantitative comparison between AMT tools.

Since ground truth MIDI is unavailable, we use:
1. Cross-tool agreement metrics (pairwise note matching)
2. Note density correlation
3. Pitch distribution KL divergence
4. Temporal coverage comparison
"""
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pretty_midi
import mir_eval
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import entropy

BASE = os.path.join(os.path.dirname(__file__), '..')
PLOT_DIR = os.path.join(BASE, 'evaluation', 'plots')

# Load all MIDI files
bp = pretty_midi.PrettyMIDI(os.path.join(BASE, 'outputs', 'basic_pitch', 'basic_pitch_output.mid'))
pyin = pretty_midi.PrettyMIDI(os.path.join(BASE, 'outputs', 'pyin_baseline', 'pyin_baseline_output.mid'))
pt = pretty_midi.PrettyMIDI(os.path.join(BASE, 'outputs', 'piano_transcription', 'piano_transcription_output.mid'))
bp_pipa = pretty_midi.PrettyMIDI(os.path.join(BASE, 'outputs', 'basic_pitch', 'bp_pipa_tuned.mid'))
improved = pretty_midi.PrettyMIDI(os.path.join(BASE, 'improvement', 'improved_basic_pitch.mid'))

tools = {
    'Basic Pitch (default)': bp,
    'Basic Pitch (pipa-tuned)': bp_pipa,
    'librosa pyin': pyin,
    'Piano Transcription': pt,
    'Improved': improved,
}

def get_notes(midi_data):
    if not midi_data.instruments:
        return np.array([]), np.array([]), np.array([])
    notes = midi_data.instruments[0].notes
    if not notes:
        return np.array([]), np.array([]), np.array([])
    pitches = np.array([n.pitch for n in notes])
    onsets = np.array([n.start for n in notes])
    offsets = np.array([n.end for n in notes])
    return pitches, onsets, offsets

# 1. Pairwise mir_eval transcription metrics
print('=' * 70)
print('Pairwise mir_eval Transcription Metrics')
print('=' * 70)
print(f'{"Pair":<40} {"Prec":>6} {"Rec":>6} {"F1":>6}')
print('-' * 70)

tool_names = list(tools.keys())
pairwise_results = {}

for i in range(len(tool_names)):
    for j in range(len(tool_names)):
        if i == j:
            continue
        ref_name = tool_names[i]
        est_name = tool_names[j]
        ref_p, ref_o, ref_off = get_notes(tools[ref_name])
        est_p, est_o, est_off = get_notes(tools[est_name])

        if len(ref_p) == 0 or len(est_p) == 0:
            continue

        # Use mir_eval.transcription
        ref_intervals = np.column_stack([ref_o, ref_off])
        est_intervals = np.column_stack([est_o, est_off])
        ref_pitches_midi = ref_p.astype(float)
        est_pitches_midi = est_p.astype(float)

        try:
            p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals, ref_pitches_midi,
                est_intervals, est_pitches_midi,
                pitch_tolerance=0.5,  # half semitone
                onset_tolerance=0.05,  # 50ms
                offset_ratio=0.2,
            )
            pairwise_results[(ref_name, est_name)] = {'precision': p, 'recall': r, 'f1': f}
            print(f'{ref_name[:18]:>18} -> {est_name[:18]:<18} {p:>6.3f} {r:>6.3f} {f:>6.3f}')
        except Exception as e:
            print(f'{ref_name[:18]:>18} -> {est_name[:18]:<18} ERROR: {e}')

# 2. Note density correlation
print('\n' + '=' * 70)
print('Note Onset Density Correlation (1s windows)')
print('=' * 70)

duration = 94.3
window = 1.0
time_bins = np.arange(0, duration, window)
density_dict = {}

for name, midi_data in tools.items():
    notes = midi_data.instruments[0].notes if midi_data.instruments else []
    density = [sum(1 for n in notes if t <= n.start < t + window) for t in time_bins]
    density_dict[name] = np.array(density)

print(f'{"Pair":<50} {"Pearson r":>10}')
print('-' * 70)
density_corr = {}
for i in range(len(tool_names)):
    for j in range(i + 1, len(tool_names)):
        n1, n2 = tool_names[i], tool_names[j]
        r = np.corrcoef(density_dict[n1], density_dict[n2])[0, 1]
        density_corr[(n1, n2)] = r
        print(f'{n1[:22]:>22} vs {n2[:22]:<22} {r:>10.3f}')

# 3. Pitch distribution KL divergence
print('\n' + '=' * 70)
print('Pitch Distribution KL Divergence')
print('=' * 70)

pitch_bins = np.arange(30, 96)
pitch_dist = {}
for name, midi_data in tools.items():
    notes = midi_data.instruments[0].notes if midi_data.instruments else []
    pitches = [n.pitch for n in notes]
    hist, _ = np.histogram(pitches, bins=pitch_bins)
    hist = hist.astype(float) + 1e-10  # avoid zeros
    pitch_dist[name] = hist / hist.sum()

print(f'{"Pair":<50} {"KL Div":>10}')
print('-' * 70)
kl_results = {}
for i in range(len(tool_names)):
    for j in range(i + 1, len(tool_names)):
        n1, n2 = tool_names[i], tool_names[j]
        kl = entropy(pitch_dist[n1], pitch_dist[n2])
        kl_results[(n1, n2)] = kl
        print(f'{n1[:22]:>22} vs {n2[:22]:<22} {kl:>10.4f}')

# 4. Temporal coverage
print('\n' + '=' * 70)
print('Temporal Coverage')
print('=' * 70)

for name, midi_data in tools.items():
    notes = midi_data.instruments[0].notes if midi_data.instruments else []
    if not notes:
        print(f'{name}: no notes')
        continue
    onsets = [n.start for n in notes]
    offsets = [n.end for n in notes]
    # Fraction of 1s windows with at least 1 note
    coverage = sum(1 for t in time_bins if any(t <= n.start < t + window for n in notes)) / len(time_bins)
    print(f'{name}: {coverage*100:.1f}% coverage, onset range [{min(onsets):.1f}s, {max(onsets):.1f}s]')

# --- Visualization ---
# Density correlation heatmap
fig, ax = plt.subplots(figsize=(8, 7))
n = len(tool_names)
corr_matrix = np.eye(n)
short_names = [n[:15] for n in tool_names]
for i in range(n):
    for j in range(i + 1, n):
        key = (tool_names[i], tool_names[j])
        if key in density_corr:
            corr_matrix[i, j] = density_corr[key]
            corr_matrix[j, i] = density_corr[key]

im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-0.2, vmax=1.0)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(short_names, fontsize=9)
for i in range(n):
    for j in range(n):
        ax.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center', fontsize=8)
plt.colorbar(im, label='Pearson r')
ax.set_title('Note Density Correlation Between Tools', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'mir_eval_density_correlation.png'), dpi=150, bbox_inches='tight')
plt.close()
print('\nSaved mir_eval_density_correlation.png')

# KL divergence heatmap
fig, ax = plt.subplots(figsize=(8, 7))
kl_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i == j:
            kl_matrix[i, j] = 0
        elif (tool_names[i], tool_names[j]) in kl_results:
            kl_matrix[i, j] = kl_results[(tool_names[i], tool_names[j])]
        elif (tool_names[j], tool_names[i]) in kl_results:
            kl_matrix[i, j] = kl_results[(tool_names[j], tool_names[i])]

im = ax.imshow(kl_matrix, cmap='YlOrRd', vmin=0, vmax=max(kl_results.values()) if kl_results else 1)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(short_names, fontsize=9)
for i in range(n):
    for j in range(n):
        ax.text(j, i, f'{kl_matrix[i,j]:.3f}', ha='center', va='center', fontsize=8)
plt.colorbar(im, label='KL Divergence')
ax.set_title('Pitch Distribution KL Divergence Between Tools', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'mir_eval_kl_divergence.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved mir_eval_kl_divergence.png')

# Pairwise F1 heatmap (using pyin as reference — broadest coverage)
fig, ax = plt.subplots(figsize=(10, 7))
# Use pairwise F1 where ref is the tool name on y-axis, est is x-axis
f1_matrix = np.full((n, n), np.nan)
for (ref, est), vals in pairwise_results.items():
    i_ref = tool_names.index(ref)
    i_est = tool_names.index(est)
    f1_matrix[i_ref, i_est] = vals['f1']

im = ax.imshow(f1_matrix, cmap='RdYlGn', vmin=0, vmax=0.6)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(short_names, fontsize=9)
for i in range(n):
    for j in range(n):
        if not np.isnan(f1_matrix[i, j]):
            ax.text(j, i, f'{f1_matrix[i,j]:.2f}', ha='center', va='center', fontsize=8)
plt.colorbar(im, label='F1 Score')
ax.set_xlabel('Estimated (columns)', fontsize=11)
ax.set_ylabel('Reference (rows)', fontsize=11)
ax.set_title('Pairwise Transcription F1 (onset_tol=50ms, pitch_tol=0.5)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'mir_eval_pairwise_f1.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved mir_eval_pairwise_f1.png')

print('\nDone!')
