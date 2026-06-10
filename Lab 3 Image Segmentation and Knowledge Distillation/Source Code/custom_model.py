import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=nn.SiLU):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            act()
        )

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()
    def forward(self, x):
        x = self.act(self.dw_bn(self.depthwise(x)))
        x = self.act(self.pw_bn(self.pointwise(x)))
        return x

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=256, dilations=(1,6,12,18)):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            if d==1:
                self.branches.append(ConvBNAct(in_ch, out_ch, k=1, p=0))
            else:
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, in_ch, 3, padding=d, dilation=d, groups=in_ch, bias=False),
                        nn.BatchNorm2d(in_ch),
                        nn.SiLU(),
                        nn.Conv2d(in_ch, out_ch, 1, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.SiLU()
                    )
                )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )
        self.project = nn.Sequential(
            nn.Conv2d(len(dilations)*out_ch + out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )
    def forward(self, x):
        size = x.shape[-2:]
        feats = [b(x) for b in self.branches]
        img_pool = F.interpolate(self.image_pool(x), size=size, mode='bilinear', align_corners=True)
        feats.append(img_pool)
        x = torch.cat(feats, dim=1)
        return self.project(x)

class UNetDecoder(nn.Module):
    def __init__(self, low_ch=24, mid_ch=40, aspp_ch=256, decoder_ch=128, num_classes=21):
        super().__init__()
        self.conv_mid = ConvBNAct(mid_ch+aspp_ch, decoder_ch)
        self.conv_low = ConvBNAct(low_ch+decoder_ch, decoder_ch)
        self.classifier = nn.Conv2d(decoder_ch, num_classes, 1)
    def forward(self, aspp_feat, mid_feat, low_feat, out_size):
        x = F.interpolate(aspp_feat, size=mid_feat.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, mid_feat], dim=1)
        x = self.conv_mid(x)
        x = F.interpolate(x, size=low_feat.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, low_feat], dim=1)
        x = self.conv_low(x)
        x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=True)
        return self.classifier(x)
    
class LightweightUNetSeg(nn.Module):
    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.features = backbone.features
        self.tap_indices = {"low": 2, "mid": 5, "high": 12}
        self.low_ch, self.mid_ch, self.high_ch = 24, 40, 576
        self.aspp = ASPP(self.high_ch, out_ch=256)
        self.decoder = UNetDecoder(
            low_ch=self.low_ch,
            mid_ch=self.mid_ch,
            aspp_ch=256,
            decoder_ch=128,
            num_classes=num_classes
        )

    def forward(self, x, return_feats=False):
        size = x.shape[-2:]
        feats = []
        out = x
        for layer in self.features:
            out = layer(out)
            feats.append(out)
        low, mid, high = feats[self.tap_indices["low"]], feats[self.tap_indices["mid"]], feats[self.tap_indices["high"]]
        aspp_feat = self.aspp(high)
        logits = self.decoder(aspp_feat, mid, low, out_size=size)

        if return_feats:
            return logits, {"low": low, "mid": mid, "high": high}
        return logits
    
if __name__ == "__main__":
    model = LightweightUNetSeg(num_classes=21, pretrained=False)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")