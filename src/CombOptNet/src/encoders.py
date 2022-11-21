from . import models, utils

from pytorch_lightning.utilities.cli import instantiate_class
import torch
import torch.nn as nn
import torch.nn.functional as F

from IPython.core.debugger import Pdb
from typing import Callable, List, Optional, Tuple, Union


# TODO: ab_encoders is now subsumed by ABEncoderList
class DisjointEncoder(nn.Module):
    def __init__(
        self,
        ab_encoders: List[nn.Module],
        c_encoder: nn.Module,
    ):
        super().__init__()
        self.ab_encoders = nn.ModuleList(ab_encoders)
        self.c_encoder = c_encoder

        self.hparams = {
            "ab_encoders": utils.hparams(ab_encoders),
            "c_encoder": utils.hparams(c_encoder),
        }

    def forward(self, x):
        a_list, b_list = zip(*[enc(x) for enc in self.ab_encoders])
        a = torch.cat(a_list, dim=-2)
        b = torch.cat(b_list, dim=-1)
        c = self.c_encoder(x)
        return a, b, c


class StaticABEncoder(nn.Module):
    def __init__(
        self,
        num_vars: int,
        num_constrs: int,
        batch_size_extractor: Callable = utils.get_tensor_shape,
        init_a: Union[Callable, dict] = nn.init.normal_,
        init_r: Union[Callable, dict] = nn.init.zeros_,
        init_o: Union[Callable, dict] = nn.init.zeros_,
        init_all_fn: Union[Callable, None] = None,
        init_all_args: dict = {},
        train_a: bool = True,
        train_r: bool = True,
        train_o: bool = True,
    ):

        super().__init__()
        self.a = nn.Parameter(
            torch.empty(num_constrs, num_vars),
            requires_grad=train_a,
        )
        self.r = nn.Parameter(
            torch.empty(num_constrs),
            requires_grad=train_r,
        )
        self.o = nn.Parameter(
            torch.empty(num_constrs, num_vars),
            requires_grad=train_o,
        )
        if init_all_fn is not None:
            init_all_fn(self.a, self.r, self.o, **init_all_args)
        else:
            for init, x in [(init_a, self.a), (init_r, self.r), (init_o, self.o)]:
                if isinstance(init, Callable):
                    init(x)
                else:
                    instantiate_class(x, init)

        self.batch_size_extractor = batch_size_extractor
        self.hparams = {
            "num_constrs": utils.hparams(num_constrs),
            # 'train_a': utils.hparams(train_a),
            # 'train_r': utils.hparams(train_r),
            # 'train_o': utils.hparams(train_o),
        }
        # for init, name in \
        #         [(init_a, 'init_a'), (init_r, 'init_r'), (init_o, 'init_o')]:
        #     if isinstance(init, dict):
        #         self.hparams[name] = init['init_args']

    def forward(self, x):
        self.b = self.r * torch.linalg.vector_norm(self.a, dim=-1) - torch.sum(
            self.a * self.o, dim=-1
        )
        batch_size = self.batch_size_extractor(x)
        # return self.a.expand(x.shape[0], -1, -1), self.b.expand(x.shape[0], -1)
        return self.a.expand(batch_size, -1, -1), self.b.expand(batch_size, -1)


class StaticCEncoder(nn.Module):
    def __init__(
        self,
        num_vars: int,
        init: Union[Callable, dict] = nn.init.ones_,
        train: bool = True,
    ):

        super().__init__()
        self.c = nn.Parameter(torch.empty(num_vars), requires_grad=train)
        if isinstance(init, Callable):
            init(self.c)
        else:
            instantiate_class(x, init)

        # self.hparams = {
        #         'train': utils.hparams(train),
        #         }
        # if isinstance(init, dict):
        #     self.hparams['init'] = init['init_args']

    def forward(self, x):
        return self.c.expand(x.shape[0], -1)


class IdentityCEncoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        return x
