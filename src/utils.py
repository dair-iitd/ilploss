import torch
import torch.nn.functional as F
import collections
from .CombOptNet.src.utils import get_tensor_shape


def masked_max(x, m, dim):
    ret = torch.max(m * x + ~m * torch.finfo().min, dim=dim).values
    nan = torch.zeros_like(ret)
    nan[~torch.any(m, dim=dim)] = float("nan")
    return ret + nan


def masked_min(x, m, dim):
    ret = torch.min(m * x + ~m * torch.finfo().max, dim=dim).values
    nan = torch.zeros_like(ret)
    nan[~torch.any(m, dim=dim)] = float("nan")
    return ret + nan


def one_hot_argmin(x, dim):
    return F.one_hot(torch.argmin(x, dim=dim), num_classes=x.shape[dim]).float()


def compute_normalized_solution(y, lb, ub):
    mean = (lb + ub) / 2
    size = ub - lb
    y_normalized = (y - mean) / size
    return y_normalized


def compute_denormalized_solution(y_normalized, lb, ub):
    mean = (ub + lb) / 2
    size = ub - lb
    y = y_normalized * size + mean
    return y


def hparams(x):
    if isinstance(x, (bool, int, float, str)):
        return x
    elif isinstance(x, collections.abc.Mapping):
        return {k: hparams(v) for k, v in x.items()}
    elif isinstance(x, collections.abc.Iterable):
        return {str(i): hparams(xx) for i, xx in enumerate(x)}
    else:
        return getattr(x, "hparams", {})


def zero_one_hot_encode(x, num_classes):
    nz = x != 0
    ret = F.one_hot(nz * (x - 1).long(), num_classes=num_classes)
    return ret * nz[..., None]


def zero_one_hot_decode(x, dim=-1):
    return (torch.argmax(x, dim=dim) + 1) * torch.any(x != 0, dim=dim)


def sudoku_to_str(n, m, x, hc="-", vc="|", cc="|", sc=" ", nc="\n"):
    g = x.view(m, n, n, m)
    mx = max([len(str(t)) for t in x.view(-1).tolist()])
    bl = []
    for b in range(m):
        rl = []
        for r in range(n):
            sl = []
            for s in range(n):
                cl = [""]
                for c in range(m):
                    cl.append(f"{g[b,r,s,c].item(): <{mx}}")
                cl.append("")
                sl.append(sc.join(cl))
            rl.append(vc.join(sl))
        bl.append(nc.join(rl))
    bc = nc + cc.join([hc.join([""] + [hc * mx] * m + [""])] * n) + nc
    return bc.join(bl)


def sudokus_to_str(n, m, xs, caps=None, pc=" ", nc="\n", **kwargs):
    t = [sudoku_to_str(n, m, x, nc=nc, **kwargs).split(nc) for x in xs]
    ws = [max(map(len, l)) for l in t]
    t = [pc.join(l) for l in zip(*t)]
    if caps is not None:
        t = [pc.join([f"{cap: ^{w}}" for w, cap in zip(ws, caps)])] + t
    t = nc.join(t)
    return t


def wrap_text(s, tag="pre"):
    return f"<{tag}>{s}</{tag}>"


def flatten_dict(d, sep="/"):
    ret = {}
    for k, v in d.items():
        if not isinstance(v, dict):
            ret[k] = v
            continue
        for kk, vv in flatten_dict(v).items():
            ret[sep.join([k, kk])] = vv
    return ret
