"""
Thin wrapper around a pretrained seq2seq model for phone-string -> text.

Loads a Hugging Face encoder-decoder (default ``allegro/plt5-small``), formats
phone strings with an optional T5 task prefix, and generates Polish text.
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .config import P2GConfig as P
from .data import format_phones
from src.utils.device import get_device


class P2GModel:
    def __init__(
        self,
        model_name: str = P.MODEL_NAME,
        device: torch.device | None = None,
        task_prefix: str = P.TASK_PREFIX,
        max_source_len: int = P.MAX_SOURCE_LEN,
        max_target_len: int = P.MAX_TARGET_LEN,
        tokenizer=None,
        model=None,
    ) -> None:
        self.device = device or get_device()
        self.task_prefix = task_prefix
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.model = (model or AutoModelForSeq2SeqLM.from_pretrained(model_name)).to(
            self.device
        )

    @classmethod
    def from_pretrained(
        cls, path: str, device: torch.device | None = None, **kwargs
    ) -> "P2GModel":
        """Load a previously fine-tuned P2G model directory."""
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSeq2SeqLM.from_pretrained(path)
        return cls(device=device, tokenizer=tokenizer, model=model, **kwargs)

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def _encode_source(self, phone_strings: list[str]) -> dict:
        sources = [self.task_prefix + s for s in phone_strings]
        enc = self.tokenizer(
            sources,
            padding=True,
            truncation=True,
            max_length=self.max_source_len,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    @torch.no_grad()
    def generate(self, phone_strings: list[str], num_beams: int = 4) -> list[str]:
        """Generate text for a batch of phone strings."""
        self.model.eval()
        enc = self._encode_source(phone_strings)
        out = self.model.generate(
            **enc, num_beams=num_beams, max_length=self.max_target_len
        )
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)

    def transcribe(self, phones: list[str] | str, num_beams: int = 4) -> str:
        """Transcribe a single utterance given phone labels (list) or a phone string."""
        phone_str = format_phones(phones) if isinstance(phones, list) else phones
        return self.generate([phone_str], num_beams=num_beams)[0]
