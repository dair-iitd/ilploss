import torch
import torch.nn.functional as F

import collections
from .constraint_handler_utils import sample_offset_constraints_numpy  


def initialize_static_constraints(a,r,o, **kwargs):
    ab,offset = sample_offset_constraints_numpy(a.size(1), a.size(0), **kwargs)
    #our constraints are of form: ax  + b >= 0 , but comboptnet gives ax + b <=0
    a.data = -torch.from_numpy(ab[:,:-1]).float()
    r.data = -torch.from_numpy(ab[:,-1]).float()
    o.data = torch.from_numpy(offset).float()


def get_tensor_shape(x, dim=0):
    return x.shape[dim]


def hparams(x):
    if isinstance(x, (bool, int, float, str)):
        return x
    elif isinstance(x, collections.abc.Mapping):
        return {k: hparams(v) for k, v in x.items()}
    elif isinstance(x, collections.abc.Iterable):
        return {str(i): hparams(xx) for i, xx in enumerate(x)}
    else:
        return getattr(x, "hparams", {})


def wrap_text(s, tag="pre"):
    return f"<{tag}>{s}</{tag}>"


def flatten_dict(d, sep="/"):
    ret = {}
    for k, v in d.items():
        if not isinstance(v, collections.abc.Mapping):
            ret[k] = v
            continue
        for kk, vv in flatten_dict(v).items():
            ret[sep.join([k, kk])] = vv
    return ret


def pretty_dict(d, sep="\n", mid=": "):
    if sep == "\n":
        mx = max([len(k) for k in d.keys()] if d else [0])
        return sep.join([f"{k:>{mx}}{mid}{v}" for k, v in d.items()])
    else:
        return sep.join([f"{k}{mid}{v}" for k, v in d.items()])
