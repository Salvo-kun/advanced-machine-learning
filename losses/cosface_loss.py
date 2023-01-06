import torch
import torch.nn.functional as F
from losses.genericface_loss import GenericFace


class CosFace(GenericFace):
    """reference1: <CosFace: Large Margin Cosine Loss for Deep Face Recognition>
       reference2: <Additive Margin Softmax for Face Verification>
    """
    def __init__(self, feat_dim, num_class, s=30., m=0.4):
        super(CosFace, self).__init__(feat_dim, num_class, s, m)

    def forward(self, x, y):
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        cos_theta = F.normalize(x, dim=1).mm(self.w)
        with torch.no_grad():
            d_theta = torch.zeros_like(cos_theta)
            d_theta.scatter_(1, y.view(-1, 1), -self.m, reduce='add')

        logits = self.s * (cos_theta + d_theta)

        return logits
