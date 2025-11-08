from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    # cross
    cross_slot_mapping: torch.Tensor | None = None
    encoder_seq_lens: int = 0
    encoder_cu_seqlens: torch.Tensor | None = None
    cross_block_tables: torch.Tensor | None = None
    cross_context_lens: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, 
                block_tables=None, cross_slot_mapping=None, encoder_seq_lens=0, encoder_cu_seqlens=None, cross_block_tables=None, cross_context_lens=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, cross_slot_mapping,
                       encoder_seq_lens, encoder_cu_seqlens, cross_block_tables, cross_context_lens)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()

