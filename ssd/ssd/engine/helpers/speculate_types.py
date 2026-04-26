from dataclasses import dataclass
import torch
from ssd.engine.sequence import Sequence
from abc import ABC, abstractmethod


@dataclass
class SpeculateResult:
    speculations: torch.Tensor
    logits_q: torch.Tensor
    cache_hits: torch.Tensor | None = None
    # MESA-only: 0=miss, 1=phase 1 (draft-sourced) hit, 2=phase 2 (proxy-sourced) hit.
    # All zeros for non-MESA paths.
    phase_source: torch.Tensor | None = None
    # MESA Phase 2 hybrid: per-row valid_k (suffix length) of the row served on this step.
    #   - draft-sourced (or full-K) hit / miss: K_long
    #   - proxy-sourced hit: K_short = K2
    # Phase 0 plumbing — engine still sets all rows to K_long until Phase 4.
    valid_k: torch.Tensor | None = None


@dataclass
class VerifyResult:
    new_suffixes: list[list[int]]
    recovery_tokens: list[int]
    eagle_acts: torch.Tensor | None = None  # Is this a tensor?


class SpeculatorBase(ABC):
    def __init__(self, lookahead: int, device: torch.device):
        self.lookahead = lookahead
        self.device = device

    @abstractmethod
    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        pass

    @abstractmethod
    def speculate(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        pass


class VerifierBase(ABC):
    def __init__(self, lookahead: int, device: torch.device):
        self.lookahead = lookahead
        self.device = device

    @abstractmethod
    def prefill(self, seqs: list[Sequence], eagle: bool = False) -> VerifyResult:
        pass

    @abstractmethod
    def verify(self, seqs: list[Sequence], speculate_result: SpeculateResult, eagle: bool = False) -> VerifyResult:
        pass
