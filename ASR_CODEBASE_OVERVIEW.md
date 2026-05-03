# ASR Codebase Overview (Polish)

## TL;DR
- Focus: phoneme-level ASR for Polish using PSD dataset (wav + TextGrid).
- Pipeline: wav -> log-mel -> sliding windows -> CRNN classifier.
- Status: data prep + model + training notebook exist; training run was interrupted; eval notebook shows checkpoint-loading mismatch.

## Repo map
- src/: core data pipeline, model, training, evaluation, plus dataset exploration (listening.py) and phoneme duration analysis (phoneme_durations.py).
- notebooks/: experiments and training.
- scripts/: utility scripts (pipeline sanity checks).
- data/: PSD dataset slices and summary stats.
- trained_models/: saved checkpoints.
- slowa_testowe/: ad-hoc test audio.

## Data / PSD dataset
- Each author folder contains ~500 sentences with .wav, .txt transcript, and .TextGrid with word and phoneme timings.
- Data slices: data/1-500, data/501-1000, data/1001-1500 (same schema).
- summary_with_102.txt shows a global phoneme duration summary across 11,468 TextGrid files: 815,357 phoneme instances, 56 unique phoneme labels, total 71,133.9 s.
- Note: data/data.txt appears binary (not readable as text). The root data/description.txt is absent.

## Feature extraction + labeling
- Constants in src/constants.py: 16k sample rate, N_FFT=400 (25 ms), hop=160 (10 ms), 128 mels, window=80 ms, shift=20 ms.
- wav_to_logmel: read wav, normalize, compute log-mel, per-mel standardization; expects 16k.
- parse_phonemes: parse TextGrid "phones" tier; optional silences_same merges "sp" into "sil".
- windows_and_labels: sliding windows over mel; label chosen by maximum overlap with phoneme; unmatched -> "oov".

## Dataset
- PhonemeWindowDataset (parsers.py): recursively finds TextGrid files, pairs with wav, builds all windows and labels in memory.
- Prints label counts; uses C.LABELS mapping (phonemes + "oov").

## Model
- CRNN (NeuralModel.py): Conv2d blocks on (mel, time) with freq-only pooling; 2-layer bidirectional LSTM; FC to class logits.
- Default hidden=64, dropout=0.2.

## Training
- Training utilities live in src/trainers.py (LR schedule, early stopping, standardized checkpoints).
- Training notebook uses src/trainers.py:
  - DATA_DIR = ../data/501-1000
  - max_files=300; 80/20 train/val split
  - optimizer: NAdam lr=1e-3
  - loss: CrossEntropyLoss(label_smoothing=0.1)
  - metric: torchmetrics.Accuracy
  - EPOCHS=15
  - save: trained_models/10epok_300files.pt (labels + config + state_dict)

## Evaluation
- evaluate_audio (src/evaluator.py): sliding windows, per-window predictions + top-k; optional alignment with TextGrid to compute window-level accuracy; collapses runs.
- load_checkpoint handles standardized checkpoint dicts and older raw state_dict files.
- Example evaluation paths used: data/501-1000/122/100022_997.wav and slowa_testowe/aaa.wav.

## Notebooks
- notebooks/Training.ipynb: end-to-end training pipeline; uses src imports and saves standardized checkpoints.
- notebooks/Results.ipynb: loads checkpoints via load_checkpoint and runs evaluate_audio examples.
- notebooks/model.ipynb: dataset exploration plus visualize word waveform, spectrogram, mel spectrogram with phoneme boundaries.
- notebooks/NeuralNetwork.ipynb: quick sanity checks for parsing and log-mel; no training.

## Known issues / inconsistencies
- Dataset loads all windows into RAM; full dataset could be large.
- src/listening.py expects directory_name like "1-500" relative to current cwd; likely needs to run from data/ or adjust path.

## Where the project was left off
- A CRNN-based phoneme classifier is implemented and wired into a notebook training pipeline.
- Training is configured for a 300-file subset from data/501-1000; a save cell exists for a 10-epoch checkpoint.
- Evaluation uses standardized load_checkpoint and runs per-window evaluation examples.
- Utility scripts for data inspection and phoneme duration statistics exist but are not integrated into a CLI or package entry point.
- scripts/check_pipeline.py provides a lightweight end-to-end sanity check.

## Suggested next actions (if you want to resume)
- Record a single "current best" checkpoint and a small evaluation script.
