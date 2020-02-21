import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class TripletNet(nn.Module):
    def __init__(self):
        super(TripletNet, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)

    def forward(self, imgs):
        """
        imgs = list(anchor_tensor, positive_tensor, negative_tensor)
        """
        self.anc = imgs[0]
        self.pos = imgs[1]
        self.neg = imgs[2]

        embedded_anc = self.resnet50(self.anc)
        embedded_pos = self.resnet50(self.pos)
        embedded_neg = self.resnet50(self.neg)

        dist_anc2pos = F.pairwise_distance(embedded_anc, embedded_pos)
        dist_anc2neg = F.pairwise_distance(embedded_anc, embedded_neg)

        return dist_anc2pos, dist_anc2neg, [embedded_anc, embedded_pos, embedded_neg]
