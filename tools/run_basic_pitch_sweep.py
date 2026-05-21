"""Run Basic Pitch with multiple parameter configurations for Phase 3 improvement experiments."""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import pretty_midi
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

AUDIO = os.path.join(os.path.dirname(__file__), '..', '火把节之夜', '火把节之夜_mono.wav')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'basic_pitch')
PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'evaluation', 'plots')

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Parameter sweep configurations
configs = {
    'default': {'onset_threshold': 0.5, 'frame_threshold': 0.3, 'minimum_note_length': 127.7},
    'low_threshold': {'onset_threshold': 0.3, 'frame_threshold': 0.2, 'minimum_note_length': 127.7},
    'high_threshold': {'onset_threshold': 0.7, 'frame_threshold': 0.5, 'minimum_note_length': 127.7},
    'short_notes': {'onset_threshold': 0.5, 'frame_threshold': 0.3, 'minimum_note_length': 50},
    'very_sensitive': {'onset_threshold': 0.2, 'frame_threshold': 0.1, 'minimum_note_length': 50},
    'pipa_tuned': {'onset_threshold': 0.35, 'frame_threshold': 0.25, 'minimum_note_length': 80},
}

results = {}
print("=" * 70)
print("Basic Pitch Parameter Sweep")
print("=" * 70)

for name, params in configs.items():
    print(f"\n--- Config: {name} ---")
    print(f"  Params: {params}")

    model_output, midi_data, note_events = predict(AUDIO, **params)

    # Save MIDI
    midi_path = os.path.join(OUT_DIR, f'bp_{name}.mid')
    midi_data.write(midi_path)

    # Analyze
    notes = midi_data.instruments[0].notes if midi_data.instruments else []
    if notes:
        pitches = [n.pitch for n in notes]
        durations = [n.end - n.start for n in notes]
        velocities = [n.velocity for n in notes]
        info = {
            'num_notes': len(notes),
            'pitch_min': min(pitches),
            'pitch_max': max(pitches),
            'pitch_range': max(pitches) - min(pitches),
            'duration_min': min(durations),
            'duration_max': max(durations),
            'duration_mean': np.mean(durations),
            'velocity_mean': np.mean(velocities),
        }
    else:
        info = {'num_notes': 0}

    results[name] = {'params': params, 'info': info, 'notes': notes}
    print(f"  Notes: {info['num_notes']}")
    if info['num_notes'] > 0:
        print(f"  Pitch: {info['pitch_min']}-{info['pitch_max']} (range: {info['pitch_range']})")
        print(f"  Duration: {info['duration_min']:.3f}-{info['duration_max']:.3f}s (mean: {info['duration_mean']:.3f}s)")

# Summary table
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Config':<18} {'Notes':>6} {'Pitch Min':>9} {'Pitch Max':>9} {'Range':>6} {'Mean Dur':>9}")
print("-" * 70)
for name, r in results.items():
    info = r['info']
    if info['num_notes'] > 0:
        print(f"{name:<18} {info['num_notes']:>6} {info['pitch_min']:>9} {info['pitch_max']:>9} {info['pitch_range']:>6} {info['duration_mean']:>9.3f}")
    else:
        print(f"{name:<18} {0:>6}")

# Generate comparison plot
fig, axes = plt.subplots(len(configs), 1, figsize=(20, 4 * len(configs)))
for idx, (name, r) in enumerate(results.items()):
    ax = axes[idx]
    notes = r['notes']
    for note in notes:
        rect = plt.Rectangle((note.start, note.pitch - 0.5), note.end - note.start, 1,
                              linewidth=0.3, edgecolor='steelblue', facecolor='steelblue', alpha=0.7)
        ax.add_patch(rect)
    ax.set_xlim(0, 95)
    ax.set_ylim(30, 90)
    params = r['params']
    ax.set_title(f'{name}: onset={params["onset_threshold"]}, frame={params["frame_threshold"]}, min_len={params["minimum_note_length"]}ms | {r["info"]["num_notes"]} notes', fontsize=11)
    ax.set_ylabel('MIDI')
    ax.grid(True, alpha=0.2)
    if idx == len(configs) - 1:
        ax.set_xlabel('Time (s)')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: pretty_midi.note_number_to_name(int(x)) if 21 <= int(x) <= 108 else ''))

plt.tight_layout()
plot_path = os.path.join(PLOT_DIR, 'basic_pitch_parameter_sweep.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved comparison plot to: {plot_path}")

# Save results as CSV
csv_path = os.path.join(OUT_DIR, 'parameter_sweep_summary.csv')
with open(csv_path, 'w') as f:
    f.write('config,onset_threshold,frame_threshold,min_note_length,num_notes,pitch_min,pitch_max,pitch_range,duration_mean\n')
    for name, r in results.items():
        p = r['params']
        info = r['info']
        if info['num_notes'] > 0:
            f.write(f'{name},{p["onset_threshold"]},{p["frame_threshold"]},{p["minimum_note_length"]},'
                    f'{info["num_notes"]},{info["pitch_min"]},{info["pitch_max"]},{info["pitch_range"]},{info["duration_mean"]:.4f}\n')
        else:
            f.write(f'{name},{p["onset_threshold"]},{p["frame_threshold"]},{p["minimum_note_length"]},0,0,0,0,0\n')
print(f"Saved summary CSV to: {csv_path}")
print("\nDone!")
