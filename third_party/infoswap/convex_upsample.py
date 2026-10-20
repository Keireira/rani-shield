"""ConvexUpsample excerpt from upstream utils.py; see NOTICE.md."""
import torch
from torch import nn
import torch.nn.functional as F

class ConvexUpsample(nn.Module):
    def __init__(self, cin, factor=2):
        super(ConvexUpsample, self).__init__()
        self.factor = factor
        self.mask = nn.Sequential(
            nn.Conv2d(cin, cin*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cin*2, factor * factor * 9, 1, padding=0))

    def get_mask(self, h):
        mask = .25 * self.mask(h)
        return mask

    def upsample(self, flow, mask):
        B, C, H, W = flow.shape
        mask = mask.view(B, 1, 9, self.factor, self.factor, H, W)
        mask = torch.softmax(mask, dim=2)
        # print(mask.shape)

        up_flow = F.unfold(8 * flow, [3, 3], padding=[1, 1])
        up_flow = up_flow.view(B, C, 9, 1, 1, H, W)
        # print(up_flow.shape)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(B, C, self.factor * H, self.factor * W)

    def forward(self, h):
        mask = self.get_mask(h)
        up_h = self.upsample(h, mask)
        return up_h

