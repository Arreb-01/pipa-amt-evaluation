"""Run Piano Transcription Inference on the pipa excerpt."""
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from piano_transcription_inference import PianoTranscription, sample_rate, load_audio

AUDIO = os.path.join(os.path.dirname(__file__), 'audio_mono.wav')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'piano_transcription')
PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'evaluation', 'plots')

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

print("=" * 60)
print("Running Piano Transcription Inference on pipa excerpt")
print("=" * 60)

# Load audio
audio, _ = load_audio(AUDIO, sr=sample_rate, mono=True)
print(f"Audio loaded: {len(audio)} samples, sr={sample_rate}")

# Run transcription
midi_path = os.path.join(OUT_DIR, 'piano_transcription_output.mid')
transcriptor = PianoTranscription(device='cpu')  # Use CPU

print("Running transcription...")
transcribed_dict = transcriptor.transcribe(audio, midi_path)

print(f"Saved MIDI to: {midi_path}")

# Analyze output
import pretty_midi
midi_data = pretty_midi.PrettyMIDI(midi_path)

print(f"\n--- Piano Transcription Output Analysis ---")
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

        # Generate piano roll
        fig, ax = plt.subplots(figsize=(16, 6))
        for note in notes:
            rect = plt.Rectangle((note.start, note.pitch - 0.5), note.end - note.start, 1,
                                  linewidth=0.5, edgecolor='purple', facecolor='purple', alpha=0.7)
            ax.add_patch(rect)
        ax.set_xlim(0, midi_data.get_end_time())
        ax.set_ylim(20, 100)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('MIDI Pitch', fontsize=12)
        ax.set_title('Piano Transcription - Output (Pipa: Night of the Torch Festival)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, pos: pretty_midi.note_number_to_name(int(x)) if 21 <= int(x) <= 108 else ''))

        plt.tight_layout()
        plot_path = os.path.join(PLOT_DIR, 'piano_transcription_pianoroll.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved piano roll to: {plot_path}")

        # Save CSV
        csv_path = os.path.join(OUT_DIR, 'piano_transcription_notes.csv')
        with open(csv_path, 'w') as f:
            f.write('start,end,pitch,velocity,duration\n')
            for note in notes:
                f.write(f'{note.start:.4f},{note.end:.4f},{note.pitch},{note.velocity},{note.end-note.start:.4f}\n')
        print(f"Saved note CSV to: {csv_path}")

print("\nDone!")
