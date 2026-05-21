"""Run Basic Pitch on denoised audio with pipa-tuned parameters."""
import os
import warnings
warnings.filterwarnings('ignore')

from basic_pitch.inference import predict
import pretty_midi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

AUDIO = os.path.join(os.path.dirname(__file__), 'audio_denoised.wav')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'basic_pitch_denoised')
PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'evaluation', 'plots')

os.makedirs(OUT_DIR, exist_ok=True)

# Pipa-tuned parameters (best from sweep)
ONSET_THRESH = 0.35
FRAME_THRESH = 0.25
MIN_NOTE_LEN = 80  # ms

print(f'Running Basic Pitch on denoised audio...')
print(f'  onset_threshold={ONSET_THRESH}, frame_threshold={FRAME_THRESH}, min_note_length={MIN_NOTE_LEN}ms')

model_output, midi_data, note_events = predict(
    AUDIO,
    onset_threshold=ONSET_THRESH,
    frame_threshold=FRAME_THRESH,
    minimum_note_length=MIN_NOTE_LEN,
)

# Save MIDI
out_midi = os.path.join(OUT_DIR, 'basic_pitch_denoised.mid')
midi_data.write(out_midi)
print(f'Saved MIDI: {out_midi}')

# Save note CSV
notes = midi_data.instruments[0].notes if midi_data.instruments else []
import csv
out_csv = os.path.join(OUT_DIR, 'basic_pitch_denoised_notes.csv')
with open(out_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['start', 'end', 'pitch', 'velocity', 'duration'])
    for n in notes:
        writer.writerow([f'{n.start:.4f}', f'{n.end:.4f}', n.pitch, n.velocity, f'{n.end-n.start:.4f}'])
print(f'Saved CSV: {out_csv}')

# Stats
if notes:
    pitches = [n.pitch for n in notes]
    durations = [n.end - n.start for n in notes]
    print(f'\nResults:')
    print(f'  Notes: {len(notes)}')
    print(f'  Pitch range: {min(pitches)}-{max(pitches)} ({pretty_midi.note_number_to_name(min(pitches))}-{pretty_midi.note_number_to_name(max(pitches))})')
    print(f'  Mean duration: {np.mean(durations):.3f}s')
    print(f'  Min duration: {np.min(durations):.3f}s')
    print(f'  Max duration: {np.max(durations):.3f}s')

# Compare with original pipa_tuned output
orig_midi = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'basic_pitch', 'bp_pipa_tuned.mid')
orig = pretty_midi.PrettyMIDI(orig_midi)
orig_notes = orig.instruments[0].notes if orig.instruments else []

print(f'\nComparison with original (non-denoised) pipa_tuned:')
print(f'  Original: {len(orig_notes)} notes')
print(f'  Denoised: {len(notes)} notes')
print(f'  Difference: {len(notes) - len(orig_notes):+d} notes')

# Piano roll comparison plot
fig, axes = plt.subplots(2, 1, figsize=(20, 8))

for ax, nlist, title, color in [
    (axes[0], orig_notes, f'Original Pipa-Tuned ({len(orig_notes)} notes)', '#E74C3C'),
    (axes[1], notes, f'Denoised Pipa-Tuned ({len(notes)} notes)', '#2ECC71'),
]:
    for note in nlist:
        rect = plt.Rectangle((note.start, note.pitch - 0.5), note.end - note.start, 1,
                              linewidth=0.3, edgecolor=color, facecolor=color, alpha=0.7)
        ax.add_patch(rect)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(0, 94.3)
    ax.set_ylim(30, 95)
    ax.set_ylabel('MIDI Pitch')
    ax.grid(True, alpha=0.2)

axes[-1].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'denoised_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved denoised_comparison.png')
print('Done!')
