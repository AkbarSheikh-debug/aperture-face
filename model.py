"""ResNet face embedding model.

    Face (3x112x112, aligned) -> ResNet body (ImageNet init) -> [1x1 reduce] -> BN -> Dropout -> Flatten
    -> FC -> BN -> 512-D -> L2 normalisation

Design notes
* The stem max-pool is removed: for 112x112 faces the stock ResNet would shrink layer4 to 4x4; without
  max-pool layer4 is 7x7, the same resolution as the ArcFace IR-ResNets that are standard in face
  recognition. Fine facial detail survives longer.
* The "output layer E" style head (BN-Dropout-FC-BN, Deng et al. 2019) keeps spatial layout (eyes vs mouth)
  instead of global average pooling, which matters for aligned faces.
* forward() only ever returns the normalised embedding. Identity classifiers live in common/losses.py and are
  thrown away after training, so cosine similarity is always computed on embeddings, never on logits.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

BACKBONES = {
    "resnet18": (torchvision.models.resnet18, torchvision.models.ResNet18_Weights.IMAGENET1K_V1, 512),
    "resnet34": (torchvision.models.resnet34, torchvision.models.ResNet34_Weights.IMAGENET1K_V1, 512),
    "resnet50": (torchvision.models.resnet50, torchvision.models.ResNet50_Weights.IMAGENET1K_V2, 2048),
}


class FaceEmbedder(nn.Module):
    def __init__(self, backbone="resnet50", emb_dim=512, pretrained=True, dropout=0.4, keep_maxpool=False,
                 input_size=112):
        super().__init__()
        ctor, weights, c = BACKBONES[backbone]
        net = ctor(weights=weights if pretrained else None)
        stem = [net.conv1, net.bn1, net.relu] + ([net.maxpool] if keep_maxpool else [])
        self.body = nn.Sequential(*stem, net.layer1, net.layer2, net.layer3, net.layer4)
        red = 512
        self.reduce = (nn.Sequential(nn.Conv2d(c, red, 1, bias=False), nn.BatchNorm2d(red), nn.PReLU(red))
                       if c > red else nn.Identity())
        with torch.no_grad():
            hw = self.body(torch.zeros(1, 3, input_size, input_size)).shape[-1]
        self.bn_feat = nn.BatchNorm2d(red)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(red * hw * hw, emb_dim)
        self.bn_emb = nn.BatchNorm1d(emb_dim)
        self.config = dict(backbone=backbone, emb_dim=emb_dim, dropout=dropout, keep_maxpool=keep_maxpool,
                           input_size=input_size)

    def forward_unnormalized(self, x):
        f = self.reduce(self.body(x))
        f = self.dropout(self.bn_feat(f)).flatten(1)
        return self.bn_emb(self.fc(f))

    def forward(self, x):
        return F.normalize(self.forward_unnormalized(x), dim=1)


def build_model(config: dict, pretrained=True) -> FaceEmbedder:
    keys = ("backbone", "emb_dim", "dropout", "keep_maxpool", "input_size")
    return FaceEmbedder(pretrained=pretrained, **{k: config[k] for k in keys if k in config})


def save_checkpoint(path, model: FaceEmbedder, extra: dict | None = None, half=True):
    sd = {k: (v.half() if half and v.is_floating_point() else v) for k, v in model.state_dict().items()}
    torch.save({"model": sd, "config": model.config, **(extra or {})}, path)


def load_checkpoint(path, device="cpu") -> tuple[FaceEmbedder, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"], pretrained=False)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ckpt["model"].items()})
    return model.to(device).eval(), ckpt


class ImageNetGAPBaseline(nn.Module):
    """Ablation baseline: frozen ImageNet ResNet, global-average-pooled features, no face training at all."""

    def __init__(self, backbone="resnet50"):
        super().__init__()
        ctor, weights, _ = BACKBONES[backbone]
        net = ctor(weights=weights)
        net.fc = nn.Identity()
        self.net = net

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


if __name__ == "__main__":
    m = FaceEmbedder("resnet50").eval()
    e = m(torch.randn(2, 3, 112, 112))
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print("embedding", tuple(e.shape), "norms", e.norm(dim=1).tolist(), f"params {n:.1f}M")
