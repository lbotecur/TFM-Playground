import numpy as np
import torch

from tfmplayground.interface import NanoTabPFNClassifier
from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def _model():
    torch.manual_seed(0)
    return NanoTabPFNModel(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=10)


def _set(model, low_memory):
    for block in model.transformer_blocks:
        block.low_memory = low_memory


def _data():
    torch.manual_seed(1)
    x = torch.randn(1, 30, 12)
    y = torch.randint(0, 3, (1, 20)).float()
    return x, y


def test_low_memory_is_identical_in_fp32():
    """Without autocast the only change is where the LayerNorms run (inside the chunks), which is exact."""
    model, (x, y) = _model().eval(), _data()
    with torch.no_grad():
        off = model((x, y), train_test_split_index=20, num_mem_chunks=4)
        _set(model, True)
        on = model((x, y), train_test_split_index=20, num_mem_chunks=4)
    torch.testing.assert_close(on, off, rtol=1e-5, atol=1e-5)


def test_low_memory_is_ignored_while_training():
    """With gradients on (training), the forward pass and the gradients are exactly the same."""
    x, y = _data()
    grads = []
    for low_memory in (False, True):
        model = _model().train()
        _set(model, low_memory)
        out = model((x, y), train_test_split_index=20)
        out.sum().backward()
        grads.append((out.detach(), [p.grad.clone() for p in model.parameters() if p.grad is not None]))
    assert torch.equal(grads[0][0], grads[1][0])
    assert all(torch.equal(a, b) for a, b in zip(grads[0][1], grads[1][1], strict=True))


def test_classifier_option_reaches_every_block():
    clf = NanoTabPFNClassifier(model=_model(), device="cpu", low_memory=True)
    assert all(block.low_memory for block in clf.model.transformer_blocks)
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 6)).astype(np.float32), np.arange(40) % 2
    proba = clf.fit(X[:30], y[:30]).predict_proba(X[30:])
    assert proba.shape == (10, 2) and np.allclose(proba.sum(axis=1), 1, atol=1e-5)
