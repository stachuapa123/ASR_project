# 🎙️ Polish ASR — Phoneme Recognition with CRNN and CTC

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

End-to-end Automatic Speech Recognition system for Polish, built from scratch in PyTorch. The project trains two complementary models on a 20-hour speech corpus, decodes phoneme sequences with custom edit-distance algorithms, and includes a GUI for live recording and dual-model comparison.

> Built as a learning project — implementing modern ASR techniques (SpecAugment, CTC loss, soft labels, beam decoding) from first principles rather than calling into pretrained pipelines.

## 📊 Results

| Model | Val Phone Accuracy | Word Accuracy (1000-word vocab) |
|---|---|---|
| CRNN (sliding window) — baseline | 70.0% | 37% |
| CRNN — with augmentation & soft labels | **74.7%** | **53.5%** |
| CTC (sequence model) — baseline | 81.9% (1-PER) | 37% |
| CTC — with augmentation | **87.4%** (PER 12.63%) | **55.5%** |

Word accuracy is measured against a 1000-word dictionary using phoneme-aware Damerau-Levenshtein matching. Test set is ~20 minutes of recorded speech containing over 1000 distinct words, evaluated against TextGrid-aligned time frames.



## 🧱 Dataset

Training data was provided by **Gdańsk University of Technology**:

- **~20 hours** of speech
- **~800,000 phonemes** across **39 distinct phonemes** (including Polish-specific sounds like `ś`, `ź`, `ć`, `dź`, `ą`, `ę`)
- **24 different speakers**
- Aligned with [Montreal Forced Aligner (MFA)](https://montreal-forced-aligner.readthedocs.io/) into `.TextGrid` files (intervals + phoneme labels)

**Additional data:**
- ~20 minutes of my own voice recordings
- Polish poem from a local poet, **"Brutus"** (thanks!) — used to expand vocabulary diversity

The dataset is too small for a transformer, so we chose architectures with strong inductive biases (CRNN, CTC).

## 🏗️ Architecture

### Model 1 — CRNN with sliding windows

```
Audio (16 kHz)
    ↓
Log-Mel Spectrogram (128 mel bins, 80ms window, 20ms shift)
    ↓
2D CNN (4 layers, MaxPool over frequency only)
    ↓
Bidirectional LSTM (hidden=64)
    ↓
Linear → 39 phoneme logits

```

Each 80ms window is independently classified — simple but provides per-frame phoneme probabilities. ~1.25M parameters.

### Model 2 — CTC (sequence)

Similar feature extractor + encoder, but trained with **Connectionist Temporal Classification** loss instead of per-window cross-entropy. Outputs variable-length phoneme sequences without explicit alignment — handles boundary windows naturally.

## 🔬 Training Techniques

What gave us the biggest gains:

### Data Augmentation (audio level)
- **Random noise** at SNR 15-30 dB (simulates different recording conditions)
- **Gain perturbation** ±6 dB (variable speaker distance from mic)

### Data Augmentation (spectrogram level — SpecAugment)
- **Frequency masking** — random horizontal bars zeroed (max 10 mel bins)
- **Time masking** — random vertical bars zeroed (max 2 frames for window model, more for CTC)

### Soft Labels (CRNN model)

Instead of one-hot labels, each window gets a **proportional distribution** of phonemes based on temporal overlap:

```
Window 0.22-0.30s contains:
  tS from 0.22-0.24s → 20ms / 80ms = 0.25
  e  from 0.24-0.30s → 60ms / 80ms = 0.75

Soft label: y = [0.0, ..., 0.25 (tS), 0.0, ..., 0.75 (e), ...]
```

Trained with KL divergence (equivalent to cross-entropy with soft targets). Helps boundary windows where two phonemes are present.

### Caching strategy

Lazy dataset with on-the-fly mel + augmentation was **catastrophically slow on Colab** (~100 min/epoch with Google Drive I/O). Solution:

1. Pre-compute 5 audio-augmented variants per file once
2. Cache as single `.pt` file (~10 GB in RAM)
3. Load directly to GPU with `map_location='cuda'` and `num_workers=0`
4. Training drops from 100 min/epoch to **~30 seconds/epoch** on RTX PRO 6000

## 🔤 Decoding Pipeline

Raw phoneme predictions are noisy — boundary windows produce duplicates, and the model confuses similar phonemes (`s`/`z`, `ś`/`sz`, `o`/`ą`). Decoding fixes this in two steps.

### Step 1 — Phonemes → Letters (Polish orthography)

Rule-based transliteration handling:
- Soft consonants (`ś` becomes `si` before vowels, `ś` elsewhere)
- Digraph phonemes (`sz`, `cz`, `dż`, `ć`)
- Nasal vowels (`ą`, `ę`)
- Identical-sounding pairs that differ orthographically (`rz`/`ż`, `h`/`ch`, `u`/`ó`)

### Step 2 — Fuzzy Word Matching

A **phoneme-aware Damerau-Levenshtein** distance against a 1000-word dictionary. Key modifications:

- **Asymmetric costs**: deletion (0.3) cheaper than insertion (0.8), because the model hallucinates extra phonemes more often than missing them
- **Phonetic similarity matrix**: substituting `s`↔`z` (voicing pair) costs 0.3, while `s`↔`k` (different category) costs 1.0
- **Context-aware deletion**: removing a phoneme whose neighbor is identical or similar is even cheaper — this handles the common case `[k, o, o, t]` → `kot` (boundary window duplicate)
- **Transposition**: swapping adjacent phonemes counts as a single operation (Damerau extension)

This is the biggest accuracy boost in the project. Classic Levenshtein: 37% word accuracy. Phoneme-aware Damerau-Levenshtein: **53-55%**.

## 🖥️ GUI

A CustomTkinter application that lets you record speech and compare both models side-by-side:

- **Manual start/stop recording** with live timer
- **Two columns** showing CRNN and CTC predictions independently
- **Pulsing recording indicator**, hotkeys (Space / Enter)
- **Cross-platform** — tested on Windows and Fedora Linux

Run with:

```bash
python gui.py
```

## 🚀 Installation

The project uses [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
git clone https://github.com/yourusername/ASR_project.git
cd ASR_project

# CPU-only install (recommended for development)
uv sync --extra cpu

# GPU install (NVIDIA)
uv sync --extra gpu
```

### Manual install with pip

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torchmetrics scipy sounddevice customtkinter tqdm matplotlib
```

## 📈 Training

Train the CRNN model:

```bash
jupyter notebook notebooks/crnn.ipynb
```

Train the CTC model:

```bash
jupyter notebook notebooks/ctc.ipynb
```

Both notebooks expect data in `data/` with paired `.wav` + `.TextGrid` files. Phoneme labels follow the custom 39-phoneme Polish set defined in `src/constants.py`.

## ⚙️ Hyperparameters

| Parameter | CRNN | CTC |
|---|---|---|
| Sample rate | 16 kHz | 16 kHz |
| FFT size | 1024 | 1024 |
| Hop length | 160 (10 ms) | 160 (10 ms) |
| Mel bins | 128 | 128 |
| Batch size | 64 | 32 |
| Optimizer | NAdam | AdamW |
| Initial LR | 1e-3 | 1e-3 |
| Scheduler | ReduceLROnPlateau | OneCycle |
| Loss | Soft Cross-Entropy (KL) | CTC |
| Epochs trained | 60 | 45 |
| Graphics Card on training | Nvidia G4 (colab) | Nvidia RTX 4070 Ti Super (16 GB VRAM  |

## 🎯 What Works, What Doesn't

**Works:**
- Clean studio speech → high accuracy
- Common Polish words present in the training distribution
- Models generalize across speakers in the original dataset

**Doesn't (yet):**
- Recordings from different microphones than the training set (distribution shift)
- Noisy environments below 10 dB SNR
- Out-of-vocabulary words — dictionary-bounded
- Real-time decoding without a language model — only the top-1 phoneme sequence is decoded

## 🔮 Future Work

- **Language model rescoring** — incorporate n-gram or transformer LM over candidate words (currently using flat dictionary scoring)
- **Common Voice Polish fine-tuning** — expand to ~120 hours, more speakers, more mic conditions
- **Streaming inference** — current pipeline processes complete recordings, not real-time chunks
- **Agentic model siri based model** — siri based model for opening desktop files

## 📚 References

- McAuliffe, M. et al. (2017). **Montreal Forced Aligner**: trainable text-speech alignment using Kaldi.
- Graves, A. et al. (2006). **Connectionist Temporal Classification**: labelling unsegmented sequence data with recurrent neural networks.
- Park, D. et al. (2019). **SpecAugment**: a simple data augmentation method for ASR.
- Smith, L. (2018). **Super-convergence**: very fast training using large learning rates (OneCycle).

## 🙏 Acknowledgements

- **Gdańsk University of Technology** — for providing the labeled speech corpus
- **Montreal Forced Aligner team** — for the alignment tooling that made phoneme-level labels possible
- **"Brutus"** — for generously sharing his poetry to expand our vocabulary diversity

## 📄 License

MIT — see `LICENSE`.

---

Built with `pytorch`, `torchaudio`, `customtkinter`, `numpy`, `scipy`. No pretrained ASR models used.
