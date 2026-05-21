# Pipa AMT Evaluation

Evaluating and improving Automatic Music Transcription (AMT) systems on traditional Chinese pipa music.

## Overview

Modern deep-learning AMT systems perform well on Western instruments, but their effectiveness on traditional Chinese instruments like the **pipa** (琵琶) is largely unverified. The pipa poses distinctive challenges — tremolos (轮指), slides (推、拉、吟、揉), harmonics, and percussive techniques — that are underrepresented in mainstream training data.

This project evaluates multiple AI-based AMT tools on an excerpt from **"Night of the Torch Festival" (《火把节之夜》)**, analyzes failure modes, and proposes improvements.

## What's Inside

### AMT Tools Surveyed

| Tool | Architecture | Instruments | Run Locally? |
|------|-------------|-------------|-------------|
| [Basic Pitch](https://github.com/spotify/basic-pitch) (Spotify) | Lightweight CNN (17K params) | Instrument-agnostic | Yes |
| [MT3](https://github.com/magenta/mt3) (Google Magenta) | T5 Transformer (seq2seq) | Multi-instrument | No (Python <3.11) |
| [Onsets and Frames](https://github.com/magenta/magenta) (Magenta) | CNN + BiLSTM / Transformer | Piano only | No (dep conflicts) |
| [Omnizart](https://github.com/Music-and-Culture-Technology-Lab/omnizart) | Multiple CNN/CRNN models | 11 orchestral instruments | No (build fails on Windows) |
| [Piano Transcription](https://github.com/qiuqiangkong/piano_transcription_inference) (ByteDance) | CNN (PANNs) + Transformer | Piano + pedals | Yes |
| [ReconVAT](https://github.com/KinWaiCheuk/ReconVAT) | U-Net + VAT (semi-supervised) | Piano, strings, woodwinds | No (dep issues) |
| [AnthemScore](https://www.lunaverus.com) (Lunaverus) | Custom CNN (ResNet-style) | Primarily piano | N/A (commercial) |
| librosa pyin | Statistical (pYIN algorithm) | Monophonic | Yes (baseline) |

### Evaluation Results

| Tool | Notes Detected | Pitch Range | Mean Duration | Coverage |
|------|---------------|-------------|---------------|----------|
| Basic Pitch | 90 | C4–C6 (MIDI 60–84) | 0.227s | 44.2% |
| librosa pyin | 246 | D♯2–B♭5 (MIDI 39–82) | 0.313s | 94.7% |
| Piano Transcription | 460 | F4–G♯6 (MIDI 65–91) | 0.425s | 62.1% |
| **Improved (ours)** | **125** | **C4–C6 (MIDI 60–84)** | **0.327s** | **69.5%** |

### Improvement Pipeline

We improved Basic Pitch's output through three strategies:

**A. Parameter tuning** — 6 configurations tested; pipa-tuned (onset=0.35, frame=0.25, min_note=80ms) gave best results.

**B. Audio pre-processing** — Spectral denoising had negligible impact (+2.6% notes), confirming the model's robustness to noise.

**C. Post-processing pipeline:**
1. **Pitch range filtering** — Constrain to pipa's playable range (MIDI 38–84)
2. **Gap filling / Tremolo merging** — Merge same-pitch notes with short gaps (ablation study shows these are functionally redundant)
3. **Pitch smoothing** — Median filter to remove isolated pitch outliers

### Quantitative Cross-Tool Analysis

Without ground truth MIDI, we measured inter-tool agreement:
- **Note density correlation:** Basic Pitch variants correlate strongly (r=0.77–0.89); pyin correlates poorly with all neural tools (r<0.35)
- **Pitch distribution KL divergence:** pyin vs Piano Transcription: 20.5 (fundamentally different pitch ranges)
- **Pairwise mir_eval F1:** Maximum 0.235 — all tools disagree on specific note placement

## Project Structure

```
├── 火把节之夜/                  # Source audio + score image
├── tools/                       # All Python scripts
│   ├── preprocess.py            # Audio preprocessing (stereo→mono)
│   ├── preprocess_denoise.py    # Spectral denoising
│   ├── run_basic_pitch.py       # Run Basic Pitch
│   ├── run_basic_pitch_sweep.py # Parameter sweep (6 configs)
│   ├── run_basic_pitch_denoised.py # Basic Pitch on denoised audio
│   ├── run_pyin_baseline.py     # librosa pyin baseline
│   ├── run_piano_transcription.py
│   ├── postprocess_improvement.py # Post-processing pipeline
│   ├── compare_all.py           # Tool comparison plots
│   ├── final_comparison.py      # Final visualization
│   ├── mir_eval_comparison.py   # Quantitative cross-tool metrics
│   └── ablation_study.py        # Pipeline ablation analysis
├── outputs/                     # Raw tool outputs (MIDI + CSV)
│   ├── basic_pitch/
│   ├── basic_pitch_denoised/
│   ├── pyin_baseline/
│   └── piano_transcription/
├── improvement/                 # Improved transcription
├── evaluation/plots/            # All visualization plots
├── report/
│   └── report.md               # Full academic report
```
```

## Key Findings

- **Training data bias is the root cause.** No publicly available AMT model has been trained on pipa or any traditional Chinese instrument.
- **Tremolo (轮指) is systematically mishandled.** Models either miss rapid repeated notes entirely or detect spurious harmonics.
- **Pitch bend techniques are invisible.** Slides and vibrato (推/拉/吟/揉) violate the discrete onset/offset assumptions of all evaluated models.
- **A pipa-aware AMT system** would need pipa-specific training data with technique annotations, continuous pitch representation, and contextual modeling of pipa performance patterns.

## Reproducing the Results

### Prerequisites

- Python 3.11
- pip

### Setup

```bash
# Core dependencies
pip install numpy librosa pretty_midi matplotlib mir_eval soundfile scipy

# AMT tools
pip install basic-pitch          # Basic Pitch (TensorFlow)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu  # PyTorch (CPU)
pip install piano-transcription-inference  # Piano Transcription

# Preprocess audio (stereo → mono)
python tools/preprocess.py

# Run each tool
python tools/run_basic_pitch.py
python tools/run_pyin_baseline.py
python tools/run_piano_transcription.py

# Parameter sweep
python tools/run_basic_pitch_sweep.py

# Audio pre-processing experiment
python tools/preprocess_denoise.py
python tools/run_basic_pitch_denoised.py

# Post-processing improvements
python tools/postprocess_improvement.py

# Generate comparison plots
python tools/compare_all.py
python tools/final_comparison.py

# Quantitative analysis
python tools/mir_eval_comparison.py
python tools/ablation_study.py
```

### Audio Source

The test audio is "Night of the Torch Festival" (《火把节之夜》) performed on pipa — a 94.3-second excerpt in WAV format (44.1kHz, 16-bit).

## References

1. Bittner, R., et al. "Basic Pitch: A Lightweight yet Accurate Audio-to-MIDI Converter with Pitch Bend Detection." ICASSP 2022. [arXiv:2203.09893](https://arxiv.org/abs/2203.09893)
2. Hawthorne, C. & Engel, J. "MT3: Multi-Task Multitrack Music Transcription." ICLR 2022. [arXiv:2111.03017](https://arxiv.org/abs/2111.03017)
3. Hawthorne, C., et al. "Onsets and Frames: Dual-Objective Piano Transcription." ISMIR 2018. [arXiv:1710.11153](https://arxiv.org/abs/1710.11153)
4. Kong, Q., et al. "High-Resolution Piano Transcription with Pedals by Regressing Onset and Offset Times." IEEE/ACM TASLP 2021. [arXiv:2010.01815](https://arxiv.org/abs/2010.01815)
5. Chang, S., et al. "Omnizart: A General Toolbox for Automatic Music Transcription." JOSS 2021. [arXiv:2106.00497](https://arxiv.org/abs/2106.00497)
6. Cheuk, K. W., Herremans, D., & Su, L. "ReconVAT: A Semi-Supervised Automatic Music Transcription Framework for Low-Resource Real-World Data." ACM MM 2021. [arXiv:2107.04954](https://arxiv.org/abs/2107.04954)

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
