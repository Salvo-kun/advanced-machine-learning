import torch
import torch.nn.functional as F
import math
from losses.genericface_loss import GenericFace


class GAMP(GenericFace):
    """ Generalized angular margin penalty-based loss
    """
    def __init__(self, feat_dim, num_class, s=30., m=(1.5, 0.5, 0.4)):
        super(GAMP, self).__init__(feat_dim, num_class, s, m)

    def forward(self, x, y):
        # weight normalization
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        # cos_theta and d_theta
        cos_theta = F.normalize(x, dim=1).mm(self.w)
        with torch.no_grad():
            m_theta = torch.acos(cos_theta.clamp(-1.+1e-5, 1.-1e-5))
            m_theta.scatter_(
                1, y.view(-1, 1), self.m[0], reduce='multiply',
            )
            k = (m_theta / math.pi).floor()
            sign = -2 * torch.remainder(k, 2) + 1  # (-1)**k
            phi_theta = sign * torch.cos(m_theta) - 2. * k

            theta_m = torch.acos(phi_theta.clamp(-1+1e-5, 1-1e-5))
            theta_m.scatter_(1, y.view(-1, 1), self.m[1], reduce='add')
            theta_m.clamp_(1e-5, 3.14159)

            d_theta = torch.zeros_like(theta_m)
            d_theta.scatter_(1, y.view(-1, 1), -self.m[2], reduce='add')

        logits = self.s * (torch.cos(theta_m) + d_theta)

        return logits

