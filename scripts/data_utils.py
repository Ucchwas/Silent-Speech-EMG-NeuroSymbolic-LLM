import pickle
import string
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from unidecode import unidecode


@dataclass
class NormState:
    mean: np.ndarray
    std: np.ndarray


class FeatureNormalizer:
    """Per-feature z-score: (x - mean) / (std + eps)"""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.state: Optional[NormState] = None

    def fit(self, arrays: List[np.ndarray]) -> None:
        cat = np.concatenate(arrays, axis=0)  # (sumT, D)
        self.state = NormState(mean=cat.mean(axis=0), std=cat.std(axis=0))

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.state is not None, "FeatureNormalizer not fitted/loaded"
        return (x - self.state.mean) / (self.state.std + self.eps)

    def save(self, path: str) -> None:
        assert self.state is not None, "FeatureNormalizer has no state to save"
        with open(path, "wb") as f:
            pickle.dump(self.state, f)

    def load(self, path: str) -> "FeatureNormalizer":
        with open(path, "rb") as f:
            self.state = pickle.load(f)
        return self


# ---- Backward-compatible helpers (used by train_emg_llm.py) ----
def load_normalizer(path: str, eps: float = 1e-8) -> FeatureNormalizer:
    """Load a FeatureNormalizer saved with FeatureNormalizer.save()."""
    return FeatureNormalizer(eps=eps).load(path)


def save_normalizer(normalizer: FeatureNormalizer, path: str) -> None:
    """Save a fitted FeatureNormalizer."""
    normalizer.save(path)


class TextTransform:
    """
    Transcript normalization (paper-faithful, Sec. III-C):
      - unidecode
      - lowercase
      - DELETE every character that is not [a-z0-9] or whitespace
      - collapse multiple spaces into one
      - strip

    Punctuation is *deleted*, not turned into a space. This matters: the
    closed-vocabulary benchmark writes times as "10:45 AM", which must
    normalize to the single token "1045 am" (as in the released reference
    transcripts), not to "10 45 am". Replacing punctuation with a space
    would split every time expression into two words, inflating both the
    lexicon and WER.

    Digits are kept as digits (the task is templated date/time prompts).

    Tokenization is character-level over:
      BASE_CHARS = [a-z][0-9] and space
      plus special tokens: <bos>, <eos>, <pad>
    """

    BASE_CHARS = string.ascii_lowercase + string.digits + " "
    VOCAB = list(BASE_CHARS) + ["<bos>", "<eos>", "<pad>"]
    BOS_IDX = len(BASE_CHARS)
    EOS_IDX = len(BASE_CHARS) + 1
    PAD_IDX = len(BASE_CHARS) + 2
    VOCAB_SIZE = len(BASE_CHARS) + 3

    # Drop anything that is neither an allowed character nor whitespace.
    _re_drop = re.compile(r"[^a-z0-9\s]+")
    _re_spaces = re.compile(r"\s+")

    def __init__(self):
        pass

    def clean(self, s: str) -> str:
        if s is None:
            return ""
        s = unidecode(str(s)).lower()
        # Delete disallowed characters (keep whitespace so word breaks survive)
        s = self._re_drop.sub("", s)
        # Collapse whitespace runs
        s = self._re_spaces.sub(" ", s).strip()
        return s

    def char_to_id(self, ch: str) -> Optional[int]:
        if not isinstance(ch, str) or len(ch) != 1:
            return None
        try:
            return self.BASE_CHARS.index(ch)
        except ValueError:
            return None

    def id_to_char(self, idx: int) -> Optional[str]:
        if not isinstance(idx, int):
            return None
        if 0 <= idx < len(self.BASE_CHARS):
            return self.BASE_CHARS[idx]
        return None

    def text_to_ids(self, s: str, add_bos_eos: bool = True) -> List[int]:
        s = self.clean(s)
        ids = [self.BASE_CHARS.index(c) for c in s if c in self.BASE_CHARS]
        return ([self.BOS_IDX] + ids + [self.EOS_IDX]) if add_bos_eos else ids

    def ids_to_text(self, ids: List[int]) -> str:
        out = []
        for i in ids:
            if i == self.EOS_IDX:
                break
            if 0 <= i < len(self.BASE_CHARS):
                out.append(self.BASE_CHARS[i])
        return "".join(out).strip()