"""Model contract: fixed 512-D, unit-norm embeddings; classifier is not part of the embedding model."""
import torch

from common.losses import ArcFaceHead, BatchHardTripletLoss, CombinedLoss
from model import FaceEmbedder


def test_embedding_shape_and_norm():
    for bb in ("resnet18", "resnet50"):
        m = FaceEmbedder(bb, pretrained=False).eval()
        e = m(torch.randn(3, 3, 112, 112))
        assert e.shape == (3, 512)
        assert torch.allclose(e.norm(dim=1), torch.ones(3), atol=1e-5)
        assert not any("classifier" in n or n.startswith("W") for n, _ in m.named_parameters())


def test_arcface_margin_increases_loss():
    torch.manual_seed(0)
    head = ArcFaceHead(16, 5, s=30, m=0.5)
    e, y = torch.randn(8, 16), torch.randint(0, 5, (8,))
    l_margin, _ = head(e, y)
    l_plain, _ = head(e, y, margin=0.0)
    assert l_margin > l_plain


def test_triplet_zero_when_separated():
    e = torch.tensor([[1.0, 0], [1.0, 0], [-1.0, 0], [-1.0, 0]])
    y = torch.tensor([0, 0, 1, 1])
    assert BatchHardTripletLoss(0.3)(e, y).item() == 0.0


def test_combined_loss_backward():
    m = FaceEmbedder("resnet18", pretrained=False)
    crit = CombinedLoss("arcface_triplet", 512, 4)
    x, y = torch.randn(8, 3, 112, 112), torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss, logs = crit(m.forward_unnormalized(x), y)
    loss.backward()
    assert {"cls", "triplet"} <= set(logs) and m.fc.weight.grad is not None
