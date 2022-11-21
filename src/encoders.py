from . import models, utils

from .CombOptNet.src.utils import get_tensor_shape
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm

import logging
from typing import Callable, List, Optional, Tuple
from IPython.core.debugger import Pdb

logger = logging.getLogger(__name__)


nonlinearity_dict = dict(
    tanh=torch.tanh, relu=torch.relu, sigmoid=torch.sigmoid, identity=lambda x: x
)


class NullABEncoder(nn.Module):
    def __init__(
        self,
        num_vars: int,
        batch_size_extractor: Callable = get_tensor_shape,
    ):
        super().__init__()

        self.register_buffer("a", torch.zeros(0, num_vars))
        self.register_buffer("b", torch.zeros(0))

        self.batch_size_extractor = batch_size_extractor

    def forward(self, x):
        bs = self.batch_size_extractor(x)
        return self.a[None, :, :].expand(bs, -1, -1), self.b[None, :].expand(bs, -1)


class ABEncoderList(nn.Module):
    def __init__(
        self,
        ab_encoders: List[nn.Module],
    ):
        super().__init__()
        self.ab_encoders = nn.ModuleList(ab_encoders)

        self.hparams = {
            "ab_encoders": utils.hparams(ab_encoders),
        }

    def forward(self, x):
        a_list, b_list = zip(*[enc(x) for enc in self.ab_encoders])
        a = torch.cat(a_list, dim=-2)
        b = torch.cat(b_list, dim=-1)
        return a, b


class ZeroABEncoder(nn.Module):
    def __init__(
        self,
        num_vars: int,
    ):
        super().__init__()

        self.register_buffer("a", torch.zeros(num_vars))
        self.register_buffer("b", torch.ones(()))

    def forward(self, x):
        bs = x.shape[0]

        return self.a[None, None, :].expand(bs, 1, -1), self.b[None, None].expand(bs, 1)


class KeypointMatchingABEncoder(nn.Module):
    def __init__(
        self,
        batch_size_extractor: Callable = get_tensor_shape,
        num_keypoints: Optional[int] = 0,
        num_keypoint_extractor: Optional[Callable] = None,
    ):
        super().__init__()

        self.num_keypoints2a = torch.nn.ParameterDict()
        self.num_keypoints2b = torch.nn.ParameterDict()
        self.batch_size_extractor = batch_size_extractor
        if num_keypoints > 0:
            a, b = self.make(num_keypoints)
            a = torch.nn.Parameter(a, requires_grad=False)
            b = torch.nn.Parameter(b, requires_grad=False)
            self.num_keypoints2a[str(num_keypoints)] = a
            self.num_keypoints2b[str(num_keypoints)] = b
            self.num_keypoints = num_keypoints
        else:
            assert num_keypoint_extractor is not None
        #
        self.num_keypoint_extractor = num_keypoint_extractor

    def make(self, num_keypoints):
        a_list = []
        b_list = []
        for i in range(num_keypoints):
            # ith row should have a 1
            arow = torch.zeros((num_keypoints, num_keypoints))
            arow[i] = 1.0
            a_list.append(arow.reshape(-1))
            b_list.append(-1.0)
            acol = torch.zeros((num_keypoints, num_keypoints))
            acol[:, i] = 1.0
            a_list.append(acol.reshape(-1))
            b_list.append(-1.0)
        #
        a = torch.stack(a_list, dim=0)
        b = torch.tensor(b_list)
        a = torch.cat([a, -a], dim=0)
        b = torch.cat([b, -b], dim=-1)
        return a, b

    def forward(self, x):
        batch_size = self.batch_size_extractor(x)
        if self.num_keypoint_extractor is None:
            num_keypoints = self.num_keypoints
        else:
            num_keypoints = self.num_keypoint_extractor(x)
            if str(num_keypoints) not in self.num_keypoints2a:
                a, b = self.make(num_keypoints)
                a = torch.nn.Parameter(a, requires_grad=False)
                b = torch.nn.Parameter(b, requires_grad=False)
                self.num_keypoints2a[str(num_keypoints)] = a
                self.num_keypoints2b[str(num_keypoints)] = b
            #
        a, b = (
            self.num_keypoints2a[str(num_keypoints)],
            self.num_keypoints2b[str(num_keypoints)],
        )
        return a.expand(batch_size, -1, -1), b.expand(batch_size, -1)


class EmptyABEncoder(nn.Module):
    def __init__(
        self,
        num_vars: int,
    ):
        super().__init__()

        self.register_buffer("a", torch.zeros(num_vars))
        self.register_buffer("b", torch.ones(()))

    def forward(self, x):
        bs = x.shape[0]
        return self.a[None, None, :].expand(bs, 0, -1), self.b[None, None].expand(bs, 0)


class SudokuABEncoder(nn.Module):
    def __init__(
        self,
        box_shape: Tuple[int, int],
    ):
        super().__init__()

        self.box_shape = box_shape

        n, m = box_shape
        nm = n * m
        self.a_ = nn.ParameterDict(
            {
                "row": nn.Parameter(
                    torch.zeros(1, 1, n, m, nm, m, n, n, m, nm), requires_grad=False
                ),
                "col": nn.Parameter(
                    torch.zeros(m, n, 1, 1, nm, m, n, n, m, nm), requires_grad=False
                ),
                "box": nn.Parameter(
                    torch.zeros(m, 1, n, 1, nm, m, n, n, m, nm), requires_grad=False
                ),
                "hot": nn.Parameter(
                    torch.zeros(m, n, n, m, 1, m, n, n, m, nm), requires_grad=False
                ),
            },
        )

        self.a = None
        self.b = None

    def make(self):
        device = self.a_["row"].device

        n, m = self.box_shape
        nm = n * m

        lb = torch.arange(m, device=device).view(-1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        lr = torch.arange(n, device=device).view(1, -1, 1, 1, 1, 1, 1, 1, 1, 1)
        ls = torch.arange(n, device=device).view(1, 1, -1, 1, 1, 1, 1, 1, 1, 1)
        lc = torch.arange(m, device=device).view(1, 1, 1, -1, 1, 1, 1, 1, 1, 1)
        lv = torch.arange(nm, device=device).view(1, 1, 1, 1, -1, 1, 1, 1, 1, 1)
        rb = torch.arange(m, device=device).view(1, 1, 1, 1, 1, -1, 1, 1, 1, 1)
        rr = torch.arange(n, device=device).view(1, 1, 1, 1, 1, 1, -1, 1, 1, 1)
        rs = torch.arange(n, device=device).view(1, 1, 1, 1, 1, 1, 1, -1, 1, 1)
        rc = torch.arange(m, device=device).view(1, 1, 1, 1, 1, 1, 1, 1, -1, 1)
        rv = torch.arange(nm, device=device).view(1, 1, 1, 1, 1, 1, 1, 1, 1, -1)

        idx = {
            "row": (ls == rs) & (lc == rc) & (lv == rv),
            "col": (lb == rb) & (lr == rr) & (lv == rv),
            "box": (lb == rb) & (ls == rs) & (lv == rv),
            "hot": (lb == rb) & (lr == rr) & (ls == rs) & (lc == rc),
        }

        ts = ["row", "col", "box", "hot"]
        num_vars = nm**3
        for t in ts:
            self.a_[t][idx[t].expand(*self.a_[t].shape)] = 1

        a = torch.cat([self.a_[t].view(-1, num_vars) for t in ts], dim=-2)
        b = -torch.ones(a.shape[-2], device=device)
        self.a = torch.cat([a, -a], dim=-2)
        self.b = torch.cat([b, -b], dim=-1)

    def forward(self, x):
        batch_size = x.shape[0]
        if self.a is None:
            self.make()
        return self.a.expand(batch_size, -1, -1), self.b.expand(batch_size, -1)


class RectifierABEncoder(nn.Module):
    def __init__(
        self,
        num_vars: int,
        num_units: int,
        batch_size_extractor: Callable = get_tensor_shape,
    ):
        super().__init__()

        self.num_units = num_units
        self.linear = nn.Linear(num_vars, num_units)

        self.batch_size_extractor = batch_size_extractor
        self.hparams = {
            "num_units": num_units,
        }

    def predict(self, y):
        out_intermediate = F.relu(self.linear(y))
        output = 1.0 - torch.sum(out_intermediate, dim=-1)
        return output

    def forward(self, x):
        device = x.device
        batch_size = self.batch_size_extractor(x)
        l = torch.arange(1, 2**self.num_units, device=device)
        r = torch.arange(self.num_units, device=device)
        enum = (l[:, None] & (2**r)) > 0
        a = -enum.float() @ self.linear.weight
        b = 1 - enum.float() @ self.linear.bias
        return a.expand(batch_size, -1, -1), b.expand(batch_size, -1)


class EqualityABEncoder(nn.Module):
    def __init__(
        self,
        ab_encoder: nn.Module,
        margin: float = 0.5,
    ):
        super().__init__()

        self.ab_encoder = ab_encoder
        self.margin = margin

        self.hparams = {
            "ab_encoder": utils.hparams(ab_encoder),
            "margin": utils.hparams(margin),
        }

    def forward(self, x):
        a, b = self.ab_encoder(x)
        return torch.cat([a, -a], dim=-2), torch.cat(
            [b + self.margin, -b + self.margin], dim=-1
        )


class LUToABEncoder(nn.Module):
    def __init__(
        self,
        lu_encoder: nn.Module,
    ):
        super().__init__()
        self.lu_encoder = lu_encoder

    def forward(self, x):
        l, u = self.lu_encoder(x)
        e = torch.eye(l.shape[-1], l.shape[-1], device=l.device)
        e = e.expand(l.shape[0], -1, -1)
        a = torch.cat([e, -e], dim=-2)
        b = torch.cat([-l, u], dim=-1)
        return a, b


class StaticLUEncoder(nn.Module):
    def __init__(
        self,
        num_vars: int,
        lb: float,
        ub: float,
        batch_size_extractor: Callable = utils.get_tensor_shape,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.batch_size_extractor = batch_size_extractor

        self.l = nn.Parameter(torch.empty(self.num_vars).fill_(lb), requires_grad=False)
        self.u = nn.Parameter(torch.empty(self.num_vars).fill_(ub), requires_grad=False)

        self.hparams = {
            "lb": utils.hparams(lb),
            "ub": utils.hparams(ub),
        }

    def forward(self, x):
        # bs = x.shape[0]
        bs = self.batch_size_extractor(x)
        return self.l[None, :].expand(bs, -1), self.u[None, :].expand(bs, -1)


class SymSudokuCEncoderForRRN(nn.Module):
    def __init__(
        self,
        num_classes: int,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hparams = {}

    def forward(self, x):
        bs = x.shape[0]
        x = x.view(bs, -1, self.num_classes)
        x = torch.cat([(x.sum(dim=-1, keepdim=True) == 0).float(), x], dim=-1)
        return x


class VisSudokuCEncoderForRRN(nn.Module):
    def __init__(
        self,
        classifier: nn.Module,
    ):
        super().__init__()
        self.classifier = classifier

        self.hparams = {
            "classifier": utils.hparams(classifier),
        }

    def forward(self, x):
        scores = self.classifier(x.view(-1, 1, 28, 28)).view(x.shape[0], x.shape[1], -1)
        probs = F.softmax(scores, dim=-1)
        # probs = torch.cat([torch.zeros(x.shape[0],x.shape[1],1).to(probs.device), probs], dim=-1)
        return probs


class NegativeCEncoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.hparams = {}

    def forward(self, x):
        return -x


class MyMultiMNISTCEncoder(nn.Module):
    def __init__(
        self,
        classifier: nn.Module,
    ):
        super().__init__()
        self.classifier = classifier

        self.hparams = {
            "classifier": utils.hparams(classifier),
        }

    def forward(self, batch):
        x = batch["query/img"].float()
        return self.classifier(x.view(-1, 1, 28, 28)).view(x.shape[0], -1)


class MultiMNISTCEncoder(nn.Module):
    def __init__(
        self,
        classifier: nn.Module,
    ):
        super().__init__()
        self.classifier = classifier

        self.hparams = {
            "classifier": utils.hparams(classifier),
        }

    def forward(self, x):
        return self.classifier(x.view(-1, 1, 28, 28)).view(x.shape[0], -1)


class LeNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
    ):
        super().__init__()

        self.lenet = nn.Sequential(
            nn.Conv2d(1, 6, 3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 6 * 6, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, images):
        batch_size = images.shape[0]
        features = self.lenet(images)

        return self.fc(features.reshape((batch_size, -1)))


class DigitCNN(nn.Module):
    def __init__(
        self,
        num_classes=10,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class VisualSudokuLogProbCEncoder(nn.Module):
    def __init__(
        self,
        classifier: nn.Module,
        num_classes: int,
        renormalize: bool = False,
    ):
        super().__init__()

        self.classifier = classifier
        self.num_classes = num_classes
        self.renormalize = renormalize

        self.hparams = {
            "classifier": utils.hparams(classifier),
        }

    def log_predictions(self, log_prob):
        max_prob, argmax = log_prob.max(dim=-1)
        models.model.scratch.update(
            {
                "sudokus": [
                    argmax.view(log_prob.shape[0], -1, self.num_classes),
                    torch.round(
                        100
                        * torch.exp(
                            max_prob.view(log_prob.shape[0], -1, self.num_classes)
                        )
                    ).long(),
                ],
                "labels": ["xhat", "prob"],
            }
        )

    def forward(self, x):
        batch_size = x.shape[0]
        num_cells = x.shape[1]
        # x_flat = x.flatten(start_dim=0, end_dim=1)
        x_flat = x.view(-1, 1, 28, 28)

        digit_guess_log_prob = self.classifier(x_flat)
        num_digits = digit_guess_log_prob.shape[1]
        digit_guess_log_prob = digit_guess_log_prob.reshape(
            batch_size, num_cells, num_digits
        )
        with torch.no_grad():
            self.log_predictions(digit_guess_log_prob)

        if self.renormalize:
            log_prob_of_non_zero = torch.log1p(
                -1.0 - torch.expm1(digit_guess_log_prob[:, :, 0:1])
            )
            digit_guess_log_prob = digit_guess_log_prob[:, :, 1:] - log_prob_of_non_zero
        else:
            digit_guess_log_prob = digit_guess_log_prob[:, :, 1:]

        return -1.0 * digit_guess_log_prob.flatten(start_dim=1, end_dim=2)


class KeyPointMatchingCEncoder(nn.Module):
    def __init__(
        self,
        backbone_params={},
        freeze: bool = False,
        pretrained_path: Optional[str] = None,
        prefix: Optional[str] = "",
    ):
        from .GraphMatching.BB_GM.model import Net

        super().__init__()
        self.freeze = freeze
        self.net = Net(**backbone_params)
        if pretrained_path is not None:
            pretrained_wts = torch.load(pretrained_path)
            # prefix = 'core.encoder.c_encoder.net.'
            net_wts = dict(
                [
                    (k[len(prefix) :], v)
                    for k, v in pretrained_wts["state_dict"].items()
                    if k.startswith(prefix)
                ]
            )
            self.net.load_state_dict(net_wts)

        if self.freeze:
            for name, param in self.net.named_parameters():
                param.requires_grad = False
        #
        self.hparams = {
            "freeze": self.freeze,
            "pretrained": pretrained_path if pretrained_path is not None else "NA",
        }

    def forward(self, x):
        cost = self.net(
            x["images"],
            x["Ps"],
            x["edges"],
            x["ns"],
            x["gt_perm_mat"],
            x.get("visualize", False),
            x.get("visualization_params", None),
        )
        # cost =  self.net(
        #                x['data_list'],
        #                x['points_gt_list'],
        #                x['edges_list'],
        #                x['n_points_gt_list'],
        #                x['perm_mat_list'],
        #                )
        bs = len(cost[0])
        cost = torch.stack(cost[0], dim=0).view(bs, -1)
        return cost.float()


class MLPCEncoder(nn.Module):
    def __init__(self, num_classes: int, mlp: torch.nn.Module):
        super().__init__()
        self.num_classes = num_classes
        self.mlp = mlp
        self.hparams = {}

    def forward(self, x):
        # Pdb().set_trace()
        bs, num_vars = x.size()
        mlp_input = x.view(bs, -1, self.num_classes)
        mlp_output = self.mlp(mlp_input)
        mlp_output = mlp_output.reshape(bs, -1)
        #
        return mlp_output


class MLP(torch.nn.Module):
    def __init__(
        self, out_features, in_features, hidden_layer_size, output_nonlinearity
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_layer_size)
        self.fc2 = nn.Linear(in_features=hidden_layer_size, out_features=out_features)
        self.output_nonlinearity_fn = nonlinearity_dict[output_nonlinearity]

    def forward(self, x):
        x = torch.relu(self.fc1(x.float()))
        x = self.fc2(x)
        x = self.output_nonlinearity_fn(x)
        return x


class KnapsackMLP(MLP):
    """
    Predicts normalized solution y (range [-0.5, 0.5])
    """

    def __init__(self, num_variables, reduced_embed_dim, embed_dim=4096, **kwargs):
        super().__init__(
            in_features=num_variables * reduced_embed_dim,
            out_features=num_variables,
            output_nonlinearity="sigmoid",
            **kwargs
        )
        self.reduce_embedding_layer = nn.Linear(
            in_features=embed_dim, out_features=reduced_embed_dim
        )

    def forward(self, x):
        bs = x.shape[0]
        x = self.reduce_embedding_layer(x.float())
        x = x.reshape(shape=(bs, -1))
        x = super().forward(x)
        y_norm = x - 0.5
        return y_norm


class KnapsackExtractWeightsCostFromEmbeddingMLP(MLP):
    """
    Extracts weights and prices of vector-embedding of Knapsack instance

    @return: torch.Tensor of shape (bs, num_variables) with negative extracted prices,
             torch.Tensor of shape (bs, num_constraints, num_variables + 1) with extracted weights and negative knapsack capacity
    """

    def __init__(
        self,
        num_constraints=1,
        embed_dim=4096,
        knapsack_capacity=1.0,
        weight_min=0.15,
        weight_max=0.35,
        cost_min=0.10,
        cost_max=0.45,
        output_nonlinearity="sigmoid",
        **kwargs
    ):
        self.num_constraints = num_constraints

        self.knapsack_capacity = knapsack_capacity
        self.weight_min = weight_min
        self.weight_range = weight_max - weight_min
        self.cost_min = cost_min
        self.cost_range = cost_max - cost_min

        super().__init__(
            in_features=embed_dim,
            out_features=num_constraints + 1,
            output_nonlinearity=output_nonlinearity,
            **kwargs
        )

    def forward(self, x):
        x = super().forward(x)
        batch_size = x.shape[0]
        cost, As = x.split([1, self.num_constraints], dim=-1)
        cost = -(self.cost_min + self.cost_range * cost[..., 0])
        As = As.transpose(1, 2)
        As = -(self.weight_min + self.weight_range * As)
        # bs = -torch.ones(batch_size, self.num_constraints).to(As.device) * self.knapsack_capacity
        bs = (
            torch.ones(batch_size, self.num_constraints).to(As.device)
            * self.knapsack_capacity
        )
        constraints = torch.cat([As, bs[..., None]], dim=-1)
        return cost, constraints


class KnapsackEncoder(nn.Module):
    def __init__(
        self,
        backbone_module_params: dict,
    ):
        super().__init__()

        self.backbone_module = KnapsackExtractWeightsCostFromEmbeddingMLP(
            **backbone_module_params
        )

    def forward(self, x):
        c, ab = self.backbone_module(x)
        a = ab[:, :, :-1]
        b = ab[:, :, -1]
        return a, b, c
