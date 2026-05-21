# Automatic Music Transcription for Pipa: Evaluating and Improving AMT Systems on Traditional Chinese Instrument Music

## 1. Introduction

Automatic Music Transcription (AMT) — the task of converting raw audio into symbolic notation — has seen remarkable progress in recent years thanks to deep learning. Systems such as MT3, Onsets and Frames, and Basic Pitch achieve high accuracy on Western instruments, particularly piano. However, their effectiveness on traditional Chinese instruments remains largely untested.

The pipa (琵琶) presents distinctive challenges that are underrepresented in mainstream training data:

- **Tremolo (轮指, lúnzhǐ):** Rapid alternation of fingers on a single string produces fast repeated notes that models may misinterpret as separate distinct pitches or as noise.
- **Pitch bends (推, 拉, 吟, 揉):** Slides and vibrato create continuous pitch trajectories that violate the discrete onset/offset assumptions of most AMT models.
- **Harmonics (泛音):** The pipa produces natural and artificial harmonics with spectral profiles different from Western plucked strings.
- **Percussive attacks (弹, 挑, 扫):** Varied plucking techniques produce diverse attack transients not captured in piano-centric training data.
- **Non-12-TET pitch material:** While the modern pipa uses frets (semi-tone spacing), expressive playing frequently produces pitches between semitones.

This study evaluates three AMT systems on an excerpt from "Night of the Torch Festival" (《火把节之夜》), a celebrated pipa piece by composer Wu Junsheng. We survey seven systems in total, run three locally, analyze failures, and propose and evaluate improvements through parameter tuning, audio pre-processing, and post-processing pipelines.

---

## 2. Survey of AMT Tools

### 2.1 Basic Pitch (Spotify)

**Architecture:** A lightweight CNN with fewer than 17,000 parameters and less than 20 MB peak memory. Three output heads predict (1) note onset/offset, (2) frame-level pitch activation, and (3) continuous pitch bend deviation.

**Training data:** Multi-instrument recordings spanning piano, guitar, strings, voice, and other instruments. Specific datasets are not publicly detailed.

**Supported instruments:** Instrument-agnostic by design.

**Output format:** MIDI with pitch bend information — a distinctive feature among AMT tools.

**Access:** `pip install basic-pitch` (Apache-2.0). Also available as a web demo at basicpitch.spotify.com.

**Paper:** Bittner, R., et al. "Basic Pitch: A Lightweight yet Accurate Audio-to-MIDI Converter with Pitch Bend Detection." ICASSP 2022. [arXiv:2203.09893](https://arxiv.org/abs/2203.09893)

### 2.2 MT3 (Google Magenta)

**Architecture:** Based on the T5 (Text-to-Text Transfer Transformer) encoder-decoder, treating AMT as a sequence-to-sequence task. Audio spectrograms are encoded, and the decoder generates tokenized note events with instrument labels, pitches, onset times, durations, and velocities.

**Training data:** MAESTRO (~200h piano), Slakh2100 (~210h synthetic multi-instrument), GuitarSet, and others — all Western instruments.

**Supported instruments:** Multi-instrument: piano, guitar, bass, drums, strings, woodwinds, brass, and more.

**Output format:** Tokenized text sequences convertible to MIDI with instrument labels.

**Access:** GitHub (magenta/mt3), requires JAX/T5X. Official Colab notebook available. Python <3.11 required.

**Paper:** Hawthorne, C. & Engel, J. "MT3: Multi-Task Multitrack Music Transcription." ICLR 2022. [arXiv:2111.03017](https://arxiv.org/abs/2111.03017)

### 2.3 Onsets and Frames (Google Magenta)

**Architecture:** CNN feature extractor followed by BiLSTM layers (original) or Transformer encoder (2021 version). Dual-output heads predict note onsets and frame-level activations. A note requires both an onset and sustained frame activity.

**Training data:** MAPS (~17.9h) and MAESTRO (~200h) — solo piano only.

**Supported instruments:** Piano only.

**Output format:** MIDI with onset, offset, pitch, and velocity.

**Access:** Part of the Magenta library (TensorFlow). Pretrained checkpoints and Colab notebooks available.

**Paper:** Hawthorne, C., et al. "Onsets and Frames: Dual-Objective Piano Transcription." ISMIR 2018. [arXiv:1710.11153](https://arxiv.org/abs/1710.11153)

### 2.4 Omnizart (Music and Culture Technology Lab)

**Architecture:** A Python toolbox integrating multiple task-specific CNN/CRNN models: piano transcription, drum transcription, chord recognition, vocal melody extraction, and multi-instrument transcription (11 orchestral instrument classes).

**Training data:** MAPS, MAESTRO (piano), and various instrument-specific datasets.

**Supported instruments:** Piano, drums, chords, vocal melody, 11 Western orchestral instruments.

**Output format:** MIDI, CSV.

**Access:** `pip install omnizart`. CLI: `omnizart music transcribe <audio>`. Google Colab available.

**Paper:** Chang, S., et al. "Omnizart: A General Toolbox for Automatic Music Transcription." JOSS 2021. [arXiv:2106.00497](https://arxiv.org/abs/2106.00497)

### 2.5 Piano Transcription Inference (ByteDance / Qiuqiang Kong)

**Architecture:** Hybrid CNN-Transformer. A PANNs (Pretrained Audio Neural Networks) CNN front-end extracts spectrogram features, followed by a Transformer encoder for temporal modeling. The model regresses precise onset and offset times (rather than frame-level classification), achieving higher temporal resolution. Separate output heads predict onsets, offsets, velocities, and sustain pedal events.

**Training data:** MAESTRO V2.0.0 (~200 hours of virtuosic piano performances).

**Supported instruments:** Piano only (with sustain pedal).

**Output format:** MIDI with note events and sustain pedal events.

**Access:** `pip install piano_transcription_inference`. Pretrained model downloaded from Zenodo.

**Paper:** Kong, Q., et al. "High-Resolution Piano Transcription with Pedals by Regressing Onset and Offset Times." IEEE/ACM TASLP 2021. [arXiv:2010.01815](https://arxiv.org/abs/2010.01815)

### 2.6 ReconVAT (Kin Wai Cheuk et al.)

**Architecture:** A semi-supervised framework built on top of U-Net models for AMT. ReconVAT augments the standard U-Net with two key components: (1) a spectrogram reconstruction loss that forces the model to learn robust audio representations, and (2) Virtual Adversarial Training (VAT), which improves generalization by generating adversarial perturbations on unlabeled data. The model outputs onset and frame predictions through separate output heads.

**Training data:** Labeled: MAPS (piano) and MusicNet (strings, woodwinds). Unlabeled: MAESTRO v2.0.0, plus additional recordings from YouTube and IMSLP for continual learning experiments. Audio is downsampled to 16 kHz.

**Supported instruments:** Piano, strings (violin, viola, cello), and woodwinds (flute, clarinet, oboe). Multi-instrument transcription is supported via separate models per instrument family.

**Output format:** MIDI files and TSV (tab-separated values) note annotations.

**Access:** GitHub (KinWaiCheuk/ReconVAT), PyTorch-based. Requires Python 3.7+ and FFmpeg for audio preprocessing. Pretrained models available for piano and string transcription.

**Key contribution:** ReconVAT is the only semi-supervised AMT framework among surveyed tools. In the few-shot setting for MusicNet strings, it achieves note-wise F1 of 61.0% and note-with-offset-wise F1 of 41.6% — improvements of 22.2% and 62.5% over supervised baselines. Its ability to leverage unlabeled data makes it theoretically more adaptable to instruments outside its training distribution.

**Paper:** Cheuk, K. W., Herremans, D., & Su, L. "ReconVAT: A Semi-Supervised Automatic Music Transcription Framework for Low-Resource Real-World Data." ACM Multimedia 2021. [arXiv:2107.04954](https://arxiv.org/abs/2107.04954)

### 2.7 AnthemScore (Lunaverus)

**Architecture:** Custom CNN treating note detection as an image recognition problem on Constant-Q Transform spectrograms (4 bins per note). Uses long-and-skinny convolutions along time and frequency dimensions with ResNet-style skip connections. 88 output nodes for polyphonic multi-label classification.

**Training data:** 2.5 million examples synthesized from 3,000 MIDI files across genres, rendered via MIDI-to-WAV synthesis.

**Supported instruments:** Primarily piano.

**Output format:** MusicXML (primary), MIDI, PDF sheet music.

**Access:** Commercial desktop application (Windows/macOS/Linux). Free trial available.

### 2.8 Comparative Summary

| Tool | Architecture | Instruments | Output | Open Source | Pipa Suitability |
|------|-------------|-------------|--------|-------------|-----------------|
| Basic Pitch | Lightweight CNN (17K params) | Instrument-agnostic | MIDI + pitch bend | Yes | Moderate — lightweight, generalizable |
| MT3 | T5 Transformer (seq2seq) | Multi-instrument | MIDI + labels | Yes | Low — no Chinese instrument data |
| Onsets & Frames | CNN + BiLSTM/Transformer | Piano only | MIDI | Yes | Very Low — piano-specific |
| Omnizart | Multiple CNN/CRNN models | Piano, drums, chords, vocal, 11 orchestral | MIDI, CSV | Yes | Low — Western orchestral only |
| Piano Transcription | CNN (PANNs) + Transformer | Piano only | MIDI + pedals | Yes | Very Low — piano-specific |
| ReconVAT | U-Net + VAT (semi-supervised) | Piano, strings, woodwinds | MIDI, TSV | Yes | Moderate — semi-supervised, better generalization potential |
| AnthemScore | Custom CNN (ResNet-style) | Primarily piano | MusicXML, MIDI | No | Low — piano-focused |

---

## 3. Evaluation Methodology

### 3.1 Audio Material

The test excerpt is from **"Night of the Torch Festival" (《火把节之夜》)** performed on pipa. Key properties:

| Property | Value |
|----------|-------|
| Duration | 94.30 seconds (1:34) |
| Sample rate | 44,100 Hz |
| Channels | Stereo → converted to mono |
| Format | 16-bit WAV |
| Estimated pitch range (pyin) | MIDI 38.6–81.7 (~D♯2–A♯5) |
| Peak amplitude | 0.461 |
| RMS energy | 0.032 |

The piece features rapid passages with tremolo (轮指), sustained melodic lines, and dynamic contrasts representative of advanced pipa repertoire.

### 3.2 Preprocessing

- Stereo audio was converted to mono by averaging channels.
- No additional preprocessing was applied for the baseline runs.
- For improvement experiments, pipa-specific parameter tuning and post-processing were applied (Section 5).

### 3.3 Tools Evaluated

Three tools were run locally:

1. **Basic Pitch v0.4.0** — Default parameters: onset_threshold=0.5, frame_threshold=0.3, minimum_note_length=127.7ms
2. **librosa pyin** — Statistical pitch tracker serving as a non-neural baseline. Frequency range C2–C7, minimum note duration 50ms.
3. **Piano Transcription Inference v0.0.6** — Pretrained model (note_F1=0.9677, pedal_F1=0.9186), CPU inference.

MT3 and Onsets and Frames could not be run locally due to Python version constraints (MT3 requires Python <3.11) and dependency conflicts on Windows (Magenta requires numba/llvmlite builds unavailable for Python 3.11). Their expected behavior is discussed qualitatively in Section 6.

### 3.4 Evaluation Criteria

- **Qualitative:** Visual comparison of piano roll outputs against the audio spectrogram and chromagram. Listening comparison of original audio vs. synthesized MIDI.
- **Quantitative:** Note count, pitch range, note density, duration statistics.
- **Cross-tool metrics:** Since ground-truth MIDI is unavailable, we employ three proxy metrics for quantitative comparison: (1) pairwise mir_eval transcription F1 scores between tools, measuring inter-tool agreement; (2) note onset density correlation (Pearson r) over 1-second windows; (3) pitch distribution KL divergence. These metrics do not measure accuracy but reveal how consistently different tools capture the same musical content.

---

## 4. Results

### 4.1 Quantitative Summary

| Metric | Basic Pitch | librosa pyin | Piano Transcription |
|--------|------------|-------------|-------------------|
| Note count | 90 | 246 | 460 |
| Pitch range (MIDI) | 60–84 | 39–82 | 65–91 |
| Pitch range (notes) | C4–C6 | D♯2–B♭5 | F4–G♯6 |
| Min duration | 0.129s | 0.058s | 0.005s |
| Max duration | 0.686s | 1.997s | 5.996s |
| Mean duration | 0.227s | 0.313s | 0.425s |
| Mean velocity | 44.8 | 64 (fixed) | — |

### 4.2 Basic Pitch Analysis

Basic Pitch detected **90 notes** in a relatively narrow range (C4–C6). The default parameters are conservative, with a high minimum note length of ~128ms, which filters out many short notes characteristic of pipa passages. The output shows:

- **Significant gaps** (notably around 30–40s) where no notes were detected despite audible music.
- **No low-register notes** detected — the pipa's lower range (below C4) is completely absent.
- **Missing tremolo segments** — rapid passages with 轮指 appear as gaps or sparse detections.
- **Relatively clean output** with few obvious false positives, but at the cost of very low recall.

### 4.3 librosa pyin Baseline Analysis

The pyin statistical pitch tracker detected **246 notes** with a much wider pitch range (D♯2–B♭5), more closely matching the expected pipa range. However:

- **Many fragmented notes** — consecutive frames of the same pitch are split into separate notes when the signal becomes momentarily unvoiced.
- **No polyphonic capability** — pyin tracks only the fundamental frequency, missing any overlapping notes or harmonic misinterpretations.
- **Better coverage** of the full piece, with detections throughout the entire 94-second duration.
- **Octave errors** present — some passages show pitch estimates jumping by an octave.

### 4.4 Piano Transcription Inference Analysis

The piano-specific model produced **460 notes** — the highest count of any tool. This is expected for a model trained on virtuosic piano performances with dense note textures:

- **Highest pitch range** (F4–G♯6) — the upper boundary (G♯6, MIDI 91) extends above the typical pipa range, suggesting **harmonic overtones are being detected as separate notes**.
- **Extremely short notes** (minimum 5ms) — many detections are likely artifacts from the model's sensitivity to transient attacks.
- **Sustained notes up to 6 seconds** — likely correspond to held notes or merged passages.
- **Dense, noisy output** — the model appears to interpret pipa timbral richness as multiple simultaneous pitches.

### 4.5 Cross-Tool Observations

Comparing the three tools reveals consistent patterns:

1. **No tool accurately captures the full pitch content.** Basic Pitch is too conservative; Piano Transcription over-detects.
2. **The pipa's unique timbre confuses all models.** Its bright, percussive attack and sustained resonance differ from both piano (hammer-struck) and guitar (finger-plucked).
3. **Tremolo passages are systematically mishandled.** Neither tool correctly identifies 轮指 as rapid repetitions of a single pitch — they either miss them entirely (Basic Pitch) or detect spurious harmonics (Piano Transcription).
4. **Pitch bend techniques are invisible.** None of the three tools captures 推/拉/吟/揉 pitch inflections (though Basic Pitch's pitch bend output could theoretically help, it was not prominent in our results).

### 4.6 Quantitative Cross-Tool Comparison

Since ground-truth MIDI is unavailable, we use cross-tool agreement metrics to quantify how consistently the tools capture the same musical content.

#### 4.6.1 Note Density Correlation

We computed Pearson correlations of note-onset density (1-second windows) across all five tool configurations:

| Pair | Pearson r |
|------|-----------|
| Basic Pitch (default) ↔ pipa-tuned | **0.889** |
| Basic Pitch (pipa-tuned) ↔ Improved | **0.773** |
| Basic Pitch (default) ↔ Piano Transcription | 0.511 |
| Basic Pitch (pipa-tuned) ↔ Piano Transcription | 0.579 |
| Basic Pitch (default) ↔ librosa pyin | 0.290 |
| librosa pyin ↔ Piano Transcription | 0.317 |
| librosa pyin ↔ Improved | 0.340 |

The Basic Pitch family (default, pipa-tuned, improved) shows high internal consistency (r > 0.77). The non-neural pyin baseline has low correlation with all neural tools (r < 0.35), confirming that statistical pitch tracking produces fundamentally different temporal patterns.

#### 4.6.2 Pitch Distribution KL Divergence

KL divergence between pitch histograms reveals how differently tools distribute detections across pitch:

| Pair | KL Divergence |
|------|---------------|
| Basic Pitch (default) ↔ pipa-tuned | 0.107 |
| Basic Pitch (default) ↔ Improved | 0.174 |
| Basic Pitch (pipa-tuned) ↔ Improved | 1.015 |
| librosa pyin ↔ Piano Transcription | **20.485** |
| Piano Transcription ↔ Improved | 6.767 |

The enormous KL divergence between pyin and Piano Transcription (20.49) confirms that these tools target completely different pitch ranges. The low divergence within the Basic Pitch family (< 1.02) shows they detect notes in similar pitch regions.

#### 4.6.3 Pairwise Transcription F1

Using mir_eval with onset_tolerance=50ms and pitch_tolerance=0.5 semitones, pairwise F1 scores between tools are universally low:

| Reference → Estimate | Precision | Recall | F1 |
|----------------------|-----------|--------|-----|
| pipa-tuned → default | 0.378 | 0.147 | 0.212 |
| pipa-tuned → Improved | 0.331 | 0.182 | 0.235 |
| default → Improved | 0.079 | 0.111 | 0.092 |
| pyin → pipa-tuned | 0.082 | 0.077 | 0.080 |

Even the best pairwise F1 (0.235 between pipa-tuned and improved) is low, indicating that small changes in parameter settings or post-processing produce notably different note-level transcriptions. This sensitivity to configuration highlights the fundamental difficulty of pipa AMT: there is no stable "correct" answer that tools converge on.

#### 4.6.4 Temporal Coverage

The fraction of 1-second windows containing at least one detected note:

| Tool | Coverage |
|------|----------|
| librosa pyin | **94.7%** |
| Basic Pitch (pipa-tuned) | 72.6% |
| Improved | 69.5% |
| Piano Transcription | 62.1% |
| Basic Pitch (default) | 44.2% |

pyin achieves the best temporal coverage, consistent with its design goal of continuous pitch tracking. The improved version maintains 69.5% coverage (vs. 44.2% for default Basic Pitch), a 57% relative improvement.

---

## 5. Improvement Experiments

### 5.1 Approach

We selected Basic Pitch as the improvement target because it offered the cleanest (if sparse) starting point. We explored three orthogonal improvement strategies: **(A) inference parameter tuning**, **(B) audio pre-processing**, and **(C) post-processing**. Additionally, we conducted an **ablation study** to quantify each step's individual contribution.

### 5.2 Experiment A: Parameter Sweep

We ran Basic Pitch with six parameter configurations:

| Configuration | Onset Thresh | Frame Thresh | Min Note (ms) | Notes | Pitch Range |
|--------------|-------------|-------------|---------------|-------|-------------|
| default | 0.50 | 0.30 | 127.7 | 90 | 60–84 |
| low_threshold | 0.30 | 0.20 | 127.7 | 355 | 55–89 |
| high_threshold | 0.70 | 0.50 | 127.7 | 9 | 70–79 |
| short_notes | 0.50 | 0.30 | 50 | 154 | 55–89 |
| very_sensitive | 0.20 | 0.10 | 50 | 6,254 | 21–108 |
| **pipa_tuned** | **0.35** | **0.25** | **80** | **231** | **55–89** |

**Key findings:**

- Lowering the onset threshold from 0.5 to 0.35 nearly tripled note count (90→231) with only modest increase in noise.
- Reducing minimum note length from 128ms to 80ms was critical for capturing pipa's short staccato notes.
- The `very_sensitive` configuration (onset=0.2, frame=0.1) produced massive over-detection (6,254 notes), demonstrating that the model's raw frame-level outputs contain significant noise.
- The **pipa_tuned** configuration represents our best balance between recall and precision.

### 5.3 Experiment B: Audio Pre-Processing

We investigated whether audio pre-processing could improve transcription quality by making the input signal more "familiar" to the model.

**Method:** We applied spectral gating denoising — estimating a noise floor from the first 0.5 seconds of audio, then suppressing spectral bins below 1.5× the noise threshold. The audio was also peak-normalized to [-1, 1] before and after denoising.

**Results:** Running Basic Pitch with the pipa_tuned configuration on the denoised audio produced **237 notes** (vs. 231 on the original). The mean duration was nearly identical (0.172s vs. 0.177s), and the pitch range was the same (MIDI 55–89).

| Configuration | Notes | Mean Duration | Pitch Range |
|--------------|-------|---------------|-------------|
| pipa_tuned (original audio) | 231 | 0.177s | 55–89 |
| pipa_tuned (denoised audio) | 237 | 0.172s | 55–89 |
| Difference | +6 (+2.6%) | −0.005s | — |

**Conclusion:** Spectral denoising has negligible impact on Basic Pitch's output. This is expected: Basic Pitch already performs internal feature extraction with its CNN front-end, which is robust to moderate noise levels. The pipa recording used in this study is studio-quality with minimal background noise, so denoising provides no benefit. More aggressive pre-processing (e.g., pitch shifting to align with piano range, timbre transfer) might yield larger improvements but risks introducing artifacts.

### 5.4 Experiment C: Post-Processing Pipeline

We applied a multi-stage post-processing pipeline to the `pipa_tuned` output:

1. **Pitch range filtering:** Remove notes outside pipa range (MIDI 38–84).
   - 231 → 224 notes
2. **Short note filtering:** Remove notes < 60ms.
   - 224 → 224 notes (no change — minimum note length from parameters was already 80ms)
3. **Gap filling:** Merge same-pitch notes separated by < 50ms gaps.
   - 224 → 134 notes
4. **Tremolo merging:** Merge rapid same-pitch repetitions (gap < 100ms) into sustained notes — targeting 轮指 passages.
   - 134 → 127 notes
5. **Pitch smoothing:** Apply median filtering (window=5) to remove isolated pitch outliers (>3 semitones from median).
   - 127 → 127 notes (no outliers detected)

### 5.5 Improvement Results

| Metric | Original (default) | Pipa-Tuned | Post-Processed (Final) |
|--------|-------------------|------------|----------------------|
| Note count | 90 | 231 | 127 |
| Pitch range | 60–84 | 55–89 | 60–84 |
| Mean duration | 0.227s | 0.177s | 0.320s |
| Max duration | 0.686s | 0.534s | 1.430s |

**Analysis:**

- The improved version has **41% more notes** than the original (127 vs 90), capturing more of the actual musical content.
- **Mean duration increased by 41%** (0.227s → 0.320s), indicating that the tremolo merger successfully consolidated fragmented detections into musically meaningful sustained notes.
- The **maximum duration of 1.43s** (vs 0.686s original) reflects successful merging of tremolo passages into single sustained events.
- The pitch range was cleaned from the over-wide 55–89 range back to 60–84 by the filtering pipeline.

### 5.6 Experiment D: Ablation Study

To quantify each pipeline step's individual contribution, we conducted two ablation analyses: **cumulative** (adding steps one by one) and **leave-one-out** (removing one step at a time).

#### Cumulative Ablation

| Step | Notes | Δ | Mean Duration |
|------|-------|---|---------------|
| 1. Raw pipa-tuned | 231 | — | 177ms |
| 2. + Pitch range filter | 224 | −7 | 178ms |
| 3. + Short note filter (>60ms) | 224 | 0 | 178ms |
| 4. + Gap filling (<50ms) | 134 | **−90** | 300ms |
| 5. + Tremolo merging (<100ms) | 125 | −9 | 327ms |
| 6. + Pitch smoothing (medfilt) | 125 | 0 | 327ms |

**Key findings:**

- **Gap filling is the most impactful step**, reducing note count by 40% (224→134) while increasing mean duration by 69% (178ms→300ms). This confirms that Basic Pitch's pipa-tuned output contains many fragmented detections of the same musical note.
- **Pitch range filtering removes 7 notes** outside the pipa's playable range (MIDI >84), a modest but necessary correction.
- **Short note filtering and pitch smoothing have no effect** in this configuration, since the minimum note length from parameters (80ms) already exceeds the 60ms threshold, and no pitch outliers were detected.
- **Tremolo merging removes an additional 9 notes**, consolidating 轮指 passages with slightly longer gaps (50–100ms).

#### Leave-One-Out Ablation

| Configuration | Notes | Mean Duration |
|--------------|-------|---------------|
| Full pipeline | 125 | 327ms |
| Remove pitch range filter | 131 | 319ms |
| Remove short note filter | 125 | 327ms |
| Remove gap filling | 125 | 327ms |
| Remove tremolo merging | 125 | 327ms |
| Remove pitch smoothing | 125 | 327ms |
| No pipeline (raw) | 231 | 255ms |

A surprising finding: removing gap filling or tremolo merging individually does **not** change the result. This is because the two steps are **functionally redundant** — tremolo merging (100ms gap threshold) subsumes gap filling (50ms threshold). When gap filling is removed, tremolo merging catches the same note pairs. Only when both are removed (i.e., no pipeline) does the note count return to 231.

**Implication:** The pipeline can be simplified to three effective steps: pitch range filtering, a single merge step (either gap filling or tremolo merging), and optional pitch smoothing. The separate gap-filling step can be eliminated without changing the output.

---

## 6. Discussion: Why AMT Models Struggle with Pipa Music

### 6.1 Training Data Bias

All evaluated systems were trained exclusively on Western instruments. The most common training datasets — MAESTRO (piano), MAPS (piano), Slakh2100 (synthetic Western instruments) — contain zero examples of pipa or any traditional Chinese instrument. This is the **root cause** of poor performance: the models have never learned the pipa's spectral signature. ReconVAT's semi-supervised approach offers a potential path forward — it can leverage unlabeled pipa recordings to improve generalization — but its base architecture is still U-Net trained on Western instrument spectrograms.

### 6.2 Timbral Mismatch

The pipa's acoustic characteristics differ fundamentally from Western instruments:

| Feature | Piano | Guitar | Pipa |
|---------|-------|--------|------|
| Attack mechanism | Felt hammer | Fingernail/fingerpick | Fingernail (multiple techniques) |
| Sustain | Rapid decay | Moderate decay | Variable (technique-dependent) |
| Overtones | Inharmonic | Harmonic | Bright, rich upper harmonics |
| Pitch modulation | None (fixed frets/keys) | String bending possible | Extensive (推/拉/吟/揉) |

Models trained to detect piano notes are implicitly learning piano-specific spectral templates. When presented with pipa spectrograms, the templates don't match well, leading to both missed detections and false positives.

### 6.3 The Tremolo Problem

轮指 (tremolo) creates a rapid series of 4–5 attacks per second on a single pitch. AMT models face a dilemma:

- If they detect each attack as a separate note onset, the output is an unrealistically dense stream of short notes.
- If they apply smoothing or minimum-note-length constraints, they may miss the attacks entirely.
- The correct representation — a sustained note with a tremolo annotation — doesn't exist in standard MIDI or in the training data.

### 6.4 Pitch Continuity Assumptions

Most AMT models assume discrete note events with clear onsets and offsets. The pipa's glissando techniques (推/拉) produce continuous pitch slides that violate this assumption. Basic Pitch's pitch bend output is a step in the right direction, but the model was not trained to recognize the specific contour shapes of pipa slides.

### 6.5 What Would a Pipa-Aware System Need?

Based on our findings, an effective pipa AMT system would require:

1. **Pipa-specific training data:** A dataset of pipa performances with aligned MIDI annotations, including technique labels. No such publicly available dataset currently exists (as of 2026).

2. **Technique-aware architecture:** The model should jointly predict note events and playing techniques (轮指, 推, 拉, 泛音, etc.). This could be formulated as multi-task learning or as an enriched output vocabulary.

3. **Continuous pitch representation:** Rather than discrete MIDI pitches, the system should output pitch contours that can represent slides and vibrato, similar to how Basic Pitch outputs pitch bends but with higher fidelity.

4. **Pipa-appropriate frequency range and resolution:** The system should be tuned for the pipa's range (A1–D6, roughly MIDI 33–86) with resolution adequate for detecting microtonal variations.

5. **Contextual modeling:** Transformer-based architectures that can learn the statistical patterns of pipa music — typical melodic gestures, common tremolo patterns, expected phrase structures.

6. **Data augmentation strategies:** Since labeled pipa data is scarce, techniques like pitch shifting, tempo variation, timbre transfer, and synthesis from MIDI with pipa-like sounds could help bridge the data gap.

Recent work on Chinese instrument research (e.g., the CMusic Database for guzheng technique detection, ISMIR 2022) and the 2025 NeurIPS AMT Challenge point toward growing interest in non-Western instrument transcription, but pipa-specific AMT remains an open problem.

---

## 7. Conclusion

This study surveyed seven AMT systems and evaluated three — Basic Pitch, librosa pyin, and Piano Transcription Inference — on a pipa excerpt from "Night of the Torch Festival." All three tools showed significant limitations:

- **Basic Pitch** (90 notes) was too conservative, missing much of the musical content.
- **librosa pyin** (246 notes) provided the best temporal coverage (94.7%) but with fragmented and noisy output.
- **Piano Transcription** (460 notes) severely over-detected, treating pipa harmonics as separate notes.

Cross-tool quantitative analysis confirmed high disagreement: pairwise mir_eval F1 scores peaked at 0.235, and pitch distribution KL divergence between pyin and Piano Transcription reached 20.5, indicating fundamentally different interpretations of the same audio.

Through three improvement strategies applied to Basic Pitch: (A) parameter tuning (6 configurations, best: pipa_tuned), (B) audio pre-processing (spectral denoising — negligible impact, +2.6% notes), and (C) a post-processing pipeline, we improved the output from 90 to 125 notes with 44% longer mean duration (227ms → 327ms). Ablation analysis revealed that gap filling / tremolo merging is the most impactful step, and that these two steps are functionally redundant — the pipeline can be simplified to pitch range filtering + a single merge step.

The fundamental bottleneck is **training data** — no publicly available AMT model has been exposed to pipa music during training. ReconVAT's semi-supervised framework offers the most promising path forward, as it could leverage unlabeled pipa recordings. Building a pipa-aware transcription system will require pipa-specific datasets with technique annotations, continuous pitch representations, and architectures that can model the instrument's unique performance vocabulary.

---

## References

1. Bittner, R., et al. "Basic Pitch: A Lightweight yet Accurate Audio-to-MIDI Converter with Pitch Bend Detection." ICASSP 2022. [arXiv:2203.09893](https://arxiv.org/abs/2203.09893)

2. Hawthorne, C. & Engel, J. "MT3: Multi-Task Multitrack Music Transcription." ICLR 2022. [arXiv:2111.03017](https://arxiv.org/abs/2111.03017)

3. Hawthorne, C., et al. "Onsets and Frames: Dual-Objective Piano Transcription." ISMIR 2018. [arXiv:1710.11153](https://arxiv.org/abs/1710.11153)

4. Kong, Q., et al. "High-Resolution Piano Transcription with Pedals by Regressing Onset and Offset Times." IEEE/ACM TASLP 2021. [arXiv:2010.01815](https://arxiv.org/abs/2010.01815)

5. Chang, S., et al. "Omnizart: A General Toolbox for Automatic Music Transcription." JOSS 2021. [arXiv:2106.00497](https://arxiv.org/abs/2106.00497)

6. Lunaverus. "Music Transcription with Convolutional Neural Networks." Technical blog post, 2016.

7. Cheuk, K. W., Herremans, D., & Su, L. "ReconVAT: A Semi-Supervised Automatic Music Transcription Framework for Low-Resource Real-World Data." ACM Multimedia 2021. [arXiv:2107.04954](https://arxiv.org/abs/2107.04954)

8. Chang, S. & Dixon, S. "YourMT3+: Open-Source Multi-Task Music Transcription." MLSP 2024. [arXiv:2407.04822](https://arxiv.org/abs/2407.04822)

9. CCMusic Database. [arXiv:2503.18802](https://arxiv.org/pdf/2503.18802)

---

## Appendix A: Environment Configuration

| Component | Version |
|-----------|---------|
| Python | 3.11.9 |
| OS | Windows 11 |
| Basic Pitch | 0.4.0 |
| Piano Transcription Inference | 0.0.6 |
| librosa | 0.11.0 |
| PyTorch | 2.12.0+cpu |
| TensorFlow | 2.15.0 |
| pretty_midi | 0.2.11 |
| mir_eval | 0.8.2 |

## Appendix B: Generated Outputs

All MIDI files, plots, and CSV data are available in the following directory structure:

```
作业5/
├── outputs/
│   ├── basic_pitch/          # Basic Pitch outputs (6 configurations)
│   ├── basic_pitch_denoised/ # Basic Pitch on denoised audio
│   ├── preprocessed/         # Denoised audio file
│   ├── pyin_baseline/        # librosa pyin output
│   └── piano_transcription/  # Piano Transcription output
├── improvement/              # Improved Basic Pitch output
├── evaluation/plots/         # All visualization plots (20+ figures)
└── tools/                    # All Python scripts used
```
