"""Config for the phone-to-text (P2G) seq2seq stage (phone set lives in CTCConfig)."""


class P2GConfig:
    # Pretrained seq2seq to fine-tune
    MODEL_NAME = "allegro/plt5-small"

    # T5 task prefix prepended to the phone string (empty disables)
    TASK_PREFIX = "fonemy na tekst: "

    # Phone-string formatting.
    KEEP_SIL = True  # keep sil/sp tokens as word-boundary cues
    PHONE_SEP = " "  # phones are joined by this separator

    # tokenizer length caps (~1 token per phone)
    MAX_SOURCE_LEN = 384
    MAX_TARGET_LEN = 256

    # True -> lowercase + strip punctuation (phones can't encode these)
    NORMALIZE_TARGET = True

    # fine-tuned models saved here (one subdir per run)
    CHECKPOINT_DIR = "trained_models"

    # Speaker-disjoint split fractions
    VAL_SPEAKER_FRAC = 0.15
    TEST_SPEAKER_FRAC = 0.15

    SEED = 42
