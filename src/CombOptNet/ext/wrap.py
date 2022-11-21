import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "CombOptNetJAX"))

from .CombOptNetJAX.models.comboptnet import CombOptNetModule
from .CombOptNetJAX.models.modules import CvxpyModule

import torch
import torch.nn as nn
import torch.nn.functional as F


# XXX: couldn't get ray to work
class CombOptNet(nn.Module):
    def __init__(
        self,
        lb: float = 0.0,
        ub: float = 1.0,
        tau: float = 1.0,
        num_threads: int = 1,
    ):
        super().__init__()

        self.net = CombOptNetModule(
            variable_range={"lb": lb, "ub": ub},
            tau=tau,
            num_threads=num_threads,
        )

        self.hparams = {
            "tau": tau,
        }

    def forward(self, a, b, c, h=None):
        norm_a = torch.linalg.vector_norm(a, dim=-1)
        a = a / norm_a[..., :, None]
        b = b / norm_a
        norm_c = torch.linalg.vector_norm(c, dim=-1)
        c = c / norm_c[..., None]
        y = self.net(c, torch.cat([-a, -b[:, :, None]], dim=-1)).float()
        status = torch.zeros(a.shape[0], device=a.device, dtype=torch.long)
        return y, status


class Cvxpy(nn.Module):
    def __init__(
        self,
        lb: float = 0.0,
        ub: float = 1.0,
        use_entropy: bool = False,
    ):
        super().__init__()

        self.net = CvxpyModule(
            variable_range={"lb": lb, "ub": ub},
            use_entropy=use_entropy,
        )

        self.hparams = {
            "use_entropy": use_entropy,
        }

    def forward(self, a, b, c, h=None):
        y = self.net(c, torch.cat([-a, -b[:, :, None]], dim=-1)).float()
        status = torch.zeros(a.shape[0], device=a.device, dtype=torch.long)
        return y, status
