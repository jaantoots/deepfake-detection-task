"""Classification model"""
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import Bottleneck, ResNet

MODEL_URLS = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def resnext50_32x4d(pretrained=False, weights=None, **kwargs):
    """Construct a ResNeXt50 model"""
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if weights is None:
        weights = model_zoo.load_url(MODEL_URLS["resnext50_32x4d"])
    if pretrained:
        # Remove mismatching layers
        del weights["fc.weight"]
        del weights["fc.bias"]
        model.load_state_dict(weights, strict=False)
    return model
