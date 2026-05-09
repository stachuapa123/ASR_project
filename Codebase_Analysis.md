# ASR Polish Phoneme Recognition - Codebase Analysis

## 1. Overview
The current workspace defines a complete deep learning pipeline using PyTorch for Automatic Speech Recognition (ASR), specifically tuned to identify Polish phonemes. It processes raw `.wav` clips, converts them to Log-Mel spectrograms, and trains a Convolutional Recurrent Neural Network (CRNN) on sliding segments of audio to classify overlapping speech phonemes annotated in Praat `.TextGrid` files.

## 2. General Directory Structure
*   **`src/`**: The core package module containing all definitions. Operations for audio parsers, window labelers, model architecture, metrics, and training loops.
*   **`data/` & `AutorskieDane/`**: Contains raw `.wav` recordings and `.TextGrid` mappings of ground truth phonetic data (start times, end times, and phonetic representation).
*   **`trained_models/`**: Repository for `.pt` and `.pth` weight checkpoints of trained and fine-tuned models.
*   **`notebooks/`**: Interactive Jupyter notebooks logging experiments regarding baseline training, augmentation usage, fine-tuning, and results evaluations.
*   **`scripts/`**: Ancillary helper scripts, containing test harnesses (e.g., `check_pipeline.py`).

## 3. End-to-End Pipeline (Current Implementation)

**A. Loading & Preprocessing**
1.  **Reading Audio & Rate Enforcement**: `src.parsers.wav_to_logmel` resamples incoming `.wav` arrays sequentially to 16 kHz to guarantee normalization across diverse recordings. Volumes are normalized to `[-1, 1]`.
2.  **Log-Mel Spectrogram Transformation**: Passing the array to Torchaudio to create a Decibel-scaled Mel Spectrogram. It uses `25 ms` windows (1024 n_fft) with `10 ms` hops.
3.  **Frame Windowing**: A classifier sliding window steps across the spectrogram. `src.constants.Constants` defines an `80 ms` window width progressing via `20 ms` shifts.
4.  **Label Mapping**: 
    *   *Hard Labeling* (`windows_and_labels`): Overlays `.TextGrid` time intervals onto the 80 ms target. The phonetic span overlapping the maximum duration of the window is set as the singular truth, and the target is indexed against 39 mapped classes.
    *   *Soft Labeling* (`windows_and_labels_soft`): An alternative processing pipeline calculates strict ratios for cases where the 80 ms window overlaps two fast phonemes.

**B. Augmentation**
Augmentation relies heavily on offline and online transformations to tackle small dataset complexities:
*   *Offline Audio Distortions*: Through `build_augmented_cache` mapping, variants with added white noise (`15-30 dB SNR`) and gain decibel scaling adjustments (`-3 to +3 dB`) are generated using `src.augment.augment_audio`.
*   *Online Spectrogram Masking*: `src.augment.SpecAugment` handles dynamic batch-level frequency & time block masking.

**C. Architecture Configuration**
The core neural model is constructed with sequential parameter reduction:
1.  **Conv Blocks**: A multi-layer 2D Convolution extracts fine spatial and frequency patterns, progressively applying `BatchNorm`, `ReLU`, and `MaxPool2d` combinations. Freuqnecy shapes are pooled but time dimension holds.
2.  **Sequential Modeling**: The convolution output passes into a dual-layer Bidirectional LSTM preserving temporal memory forwards & backward.
3.  **Classification**: Fully Connected (Linear) layers converge the RNN memory cells down to 39 nodes representing the probabilities of the Polish classes defined in `src/constants.py`.

**D. Training**
Configured to operate effectively with callbacks (`src.trainers.train_model`), incorporating PyTorch tracking (e.g., loss plateaus via `ReduceLROnPlateau`), early stopping protections, scaling metric assessments, gradient clipping, checkpoint dumping of `best_epoch`, and soft-distribution Kullback-Leibler (KL) calculation variants (`SoftCrossEntropyLoss`).

**E. Evaluation**
`src.evaluator.evaluate_audio` operates an inference iteration on an absolute audio file. The sliding window constructs a matrix of likelihood arrays predicting sequential boundaries of phonemes while evaluating average confidences & displaying aligned sequence breakdowns against `.TextGrid` truth tables if available.

---

## 4. Source File Details

### `src/constants.py`
Configuration parameters centralized. Defines:
*   Global samplerate (16000 Hz) and Fast Fourier attributes.
*   Classification sliding window parameters representing overlapping `80 ms` (size) at `20 ms` (stride).
*   Thirty-eight literal Polish phonetic tags (plus generic spaces / `oov` tags).
*   Instantiations of global transformers mapped from Torch (`MelSpectrogram` & `AmplitudeToDB`).

### `src/parsers.py`
The Data Preprocessor library.
*   `parse_phonemes()`: Takes `.TextGrid` plaintext as input, extracting (start_time, end_time, phoneme) interval boundaries.
*   `wav_to_logmel()`: Casts integer arrays into continuous 32-bit floats, corrects to 16 kHz, scales volume, computes log-mels, and standardization.
*   `windows_and_labels()` & `windows_and_labels_soft()`: Core aligners associating frames generated by logic blocks to phoneme arrays yielding strict integers or soft distribution tensors.
*   `build_audio_cache()`, `build_augmented_cache()`, & `build_augmented_cache_soft()`: Aggregating pipeline tools generating massive structured PyTorch `.pt` datasets scaling the datasets out geometrically to cover offline noise operations.
*   `PhonemeWindowDataset` & `AugmentedCacheDataset`: PyTorch Datasets.

### `src/augment.py`
Environmental interference & Regularization generators.
*   `augment_audio()`: Simulates Gaussian noise power overlaps and Decibel modifications using raw signal energy modifications.
*   `SpecAugment()`: PyTorch transform wrapper zeroing-out bands along arbitrary random axes on frequency & time planes over the spectrogram matrix.

### `src/NeuralModel.py`
*   `CRNN`: The implementation object bridging Conv2d -> Bi-LSTM -> Linear projection layers generating mapping embeddings against dimension inputs corresponding to `src.constants.Constants`.

### `src/trainers.py`
Pipeline loop management.
*   `save_checkpoint()` & `load_checkpoint()`: Utilities serializing `.state_dict()`s concurrently with evaluation metrics.
*   `train_model()`: Handles standard gradient steps. Operates on a single loader, executes LR scheduling logic plateaus, manages early stopping, and records history.
*   `evaluate_tm()`: Handles model forward passes, tracking valid outputs across `torchmetrics` instances.
*   `SoftCrossEntropyLoss()`: A mathematical extension comparing non-trivial overlapping bounds allowing the classifier to fit dual probabilities representing intersecting temporal sounds linearly. 
*   `SoftAccuracy()`: Custom torch metric logic collapsing soft distributions uniformly prior to comparisons iteratively against Argmax arrays.

### `src/evaluator.py`
*   `evaluate_audio()`: Executes the model in `no_grad` inference state sequentially across dynamic windows of new external data calculating confidences. Tracks per-window classifications versus top-candidate clusters & outputs human-readable console feedback collapses counting total model accuracy iterations iteratively.

---

## 5. Jupyter Notebooks Analysis

The `notebooks/` directory tracks interactive experiments, model prototyping, and evaluation executions. These sandbox environments operate as the primary driver for invoking the backend Python packages defined in `src/`.

### Experimentation & Workflows
*   **`Training.ipynb`**: The baseline execution script. Contains the standard routine for generating caches of unaugmented audio windows, initializing the base `CRNN` architecture, fitting it against `train_model()`, and outputting standard checkpoints.
*   **`TrainingAugment.ipynb`**: Extends the standard training loop by integrating offline environmental distortions (noise and volume offsets) dynamically throughout dataset preprocessing. Utilizes `build_augmented_cache()` arrays to stretch model resilience.
*   **`TrainingSoft.ipynb`**: Implementation of fractional distribution techniques mapping to phonetic overlaps natively within window cuts (`SoftCrossEntropyLoss`). Handles overlapping boundaries utilizing `windows_and_labels_soft()`.
*   **`FineTuning.ipynb`**: Secondary optimization loops specifically focused on updating established pre-trained weights `.pt/pth` distributions onto novel bespoke datasets (like ones aggregated in `AutorskieDane/`). Operates typically with heavily constrained base architectures or scaled-back learning rates.
*   **`NeuralNetwork.ipynb` / `model.ipynb`**: Prototyping and scratchpad environments for quickly validating tensor logic operations, model block geometries (inspecting convolutions vs recurrence arrays), and forward-pass gradient behavior inside isolated instances.
*   **`Results.ipynb`**: Responsible for running quantitative inference checks against fully baked models. It loads trained baseline/augmented files from `trained_models/` and visualizes evaluation metrics, model accuracy comparisons, and sequential outputs using `evaluate_audio()`.