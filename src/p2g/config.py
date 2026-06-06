"""
Configuration for the phone-to-text (P2G) seq2seq stage.

Mirrors the plain-class style of ``src/ctc/config.py``. The acoustic phone set
itself lives in ``CTCConfig``; here we only configure the text model that maps a
phone string to Polish text.
"""


class P2GConfig:
    # Pretrained seq2seq to fine-tune.
    MODEL_NAME = "allegro/plt5-small"

    # Optional T5 task prefix prepended to the phone string (empty string disables it).
    TASK_PREFIX = "fonemy na tekst: "

    # Phone-string formatting.
    KEEP_SIL = True  # keep sil/sp tokens as word-boundary cues
    PHONE_SEP = " "  # phones are joined by this separator

    # Tokenizer length caps (phone strings are long: ~1 token per phone for byte/sp models).
    MAX_SOURCE_LEN = 384
    MAX_TARGET_LEN = 256

    # Target text handling. False -> keep original case + punctuation (true transcription).
    NORMALIZE_TARGET = False

    # Where fine-tuned models are saved (one subdir per run).
    CHECKPOINT_DIR = "trained_models"

    # Speaker-disjoint split fractions (by speaker, not by utterance).
    VAL_SPEAKER_FRAC = 0.15
    TEST_SPEAKER_FRAC = 0.15

    SEED = 42
