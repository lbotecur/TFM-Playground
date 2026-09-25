import torch

from tfmplayground.interface import NanoTabPFNClassifier, NanoTabPFNRegressor
from tfmplayground.models.nanotabpfn import NanoTabPFNModel


def _model(num_outputs):
    torch.manual_seed(0)
    return NanoTabPFNModel(
        embedding_size=16, num_attention_heads=2, mlp_hidden_size=32, num_layers=2, num_outputs=num_outputs
    )


def test_classifier_device_is_torch_device_by_default():
    """With device=None the stored device is a torch.device, not a bare string."""
    clf = NanoTabPFNClassifier(model=_model(3), device=None)
    assert isinstance(clf.device, torch.device)


def test_classifier_device_normalizes_string():
    clf = NanoTabPFNClassifier(model=_model(3), device="cpu")
    assert isinstance(clf.device, torch.device)
    assert clf.device == torch.device("cpu")


def test_classifier_preserves_torch_device_input():
    d = torch.device("cpu")
    clf = NanoTabPFNClassifier(model=_model(3), device=d)
    assert clf.device == d


def test_regressor_device_normalizes_string():
    reg = NanoTabPFNRegressor(model=_model(8), device="cpu")
    assert isinstance(reg.device, torch.device)
    assert reg.device == torch.device("cpu")