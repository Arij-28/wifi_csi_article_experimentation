from wificsi_exp.models.network import FullModel
import torch


def test_forward():
    model = FullModel(num_classes=6)
    x = torch.randn(2, 3, 114, 500)
    out = model(x)
    assert out["logits"].shape == (2, 6)
    assert out["embeddings"].shape[0] == 2
