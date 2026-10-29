"""
test_multi_query_pool.py
Automated tests for MultiQueryAttentionPool.
"""

import torch
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("multi_query_pool", Path(__file__).parent.parent / "src" / "model" / "multi_query_attention_pool.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MultiQueryAttentionPool = _mod.MultiQueryAttentionPool


def _make_model():
    torch.manual_seed(0)
    return MultiQueryAttentionPool(dim=4, n_slot=6, n_out=12, hidden=16)


def test_output_shape_is_batch_by_n_out_not_squared():
    model = _make_model()
    x = torch.randn(3, 6, 4)
    mask = torch.ones(3, 6)
    logits, _ = model(x, mask)
    assert logits.shape == (3, 12)


def test_attention_shape_and_sums_to_one():
    model = _make_model()
    x = torch.randn(3, 6, 4)
    mask = torch.ones(3, 6)
    _, att = model(x, mask)
    assert att.shape == (3, 12, 6)
    assert torch.allclose(att.sum(-1), torch.ones(3, 12), atol=1e-5)


def test_masked_slots_get_near_zero_attention_for_all_outputs():
    model = _make_model()
    x = torch.randn(1, 6, 4)
    mask = torch.ones(1, 6)
    mask[0, -2:] = 0.0
    _, att = model(x, mask)
    assert (att[0, :, -2:] < 1e-3).all()


def test_gradients_flow_to_all_new_params():
    model = _make_model()
    x = torch.randn(3, 6, 4)
    mask = torch.ones(3, 6)
    logits, _ = model(x, mask)
    loss = logits.sum()
    loss.backward()
    assert model.query.grad is not None
    assert model.slot_emb.grad is not None
