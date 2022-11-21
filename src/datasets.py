from . import utils

from .CombOptNet.src import datasets

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST

from collections import defaultdict
import logging
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)

from IPython.core.debugger import Pdb

class SudokuDataModule(datasets.BaseDataModule):
    def __init__(
        self,
        data_path: str,
        splits: dict,
        num_validate_train: int,
        batch_sizes: dict,
        num_workers: int = 0,
        mnist_dir: Optional[str] = None,
        read_satnet_data_path: Optional[str] = None, 
        write_satnet_data_path: Optional[str] = None, 
    ):
        super().__init__(
            data_path=data_path,
            splits=splits,
            num_validate_train=num_validate_train,
            batch_sizes=batch_sizes,
            num_workers=num_workers,
        )
        self.mnist_dir = mnist_dir
        self.read_satnet_data_path = read_satnet_data_path

    def setup(self, stage=None):
        logger.info(f"load {self.data_path}...")
        pt = torch.load(self.data_path, map_location="cpu")
        logger.info("loaded.")

        full = defaultdict(dict)
        full["query"]["num"] = pt["x"].float()
        full["target"]["num"] = pt["y"].float()

        assert len(full["query"]["num"]) == len(full["target"]["num"])
        assert (
            len(full["query"]["num"])
            >= self.splits["train"] + self.splits["val"] + self.splits["test"]
        )

        n, m = pt["meta"]["box_shape"]
        num_classes = n * m
        for q in ["query", "target"]:
            full[q]["bit"] = torch.flatten(
                utils.zero_one_hot_encode(full[q]["num"], num_classes=num_classes),
                start_dim=-2,
            ).float()

        data = defaultdict(lambda: defaultdict(dict))

        for p in ["train", "val", "test"]:
            data[p]["idx"] = torch.arange(self.splits[p])
            data[p]["num_classes"] = torch.empty(self.splits[p]).fill_(num_classes)

        for q in ["query", "target"]:
            for r in ["num", "bit"]:
                data["train"][q][r] = full[q][r][: self.splits["train"]]
                data["val"][q][r] = full[q][r][
                    self.splits["train"] : self.splits["train"] + self.splits["val"]
                ]
                data["test"][q][r] = full[q][r][-self.splits["test"] :]

        if self.mnist_dir is not None:
            digit_img = {}

            mnist_dev = MNIST(self.mnist_dir, train=True, download=True)
            #mnist_data_dev = mnist_dev.data
            mnist_data_dev = (mnist_dev.data.float() / 255 - 0.1307) / 0.3081
            tmp = [mnist_data_dev[mnist_dev.targets == c] for c in range(10)]
            mn = min([len(l) for l in tmp])
            digit_img_dev = torch.stack([l[:mn] for l in tmp])[..., None, :, :]
            digit_img["train"], digit_img["val"] = torch.split(
                digit_img_dev, [int(0.8 * mn), mn - int(0.8 * mn)], 1
            )

            mnist_test = MNIST(self.mnist_dir, train=False, download=True)
            #mnist_data_test = mnist_test.data
            mnist_data_test = (mnist_test.data.float() / 255 - 0.1307) / 0.3081
            tmp = [mnist_data_test[mnist_test.targets == c] for c in range(10)]
            mn = min([len(l) for l in tmp])
            digit_img["test"] = torch.stack([l[:mn] for l in tmp])[..., None, :, :]

            for p in ["train", "val", "test"]:
                for q in ["query", "target"]:
                    rand = torch.rand(
                        data[p][q]["num"].shape,
                        generator=torch.Generator().manual_seed(0x0123_4567_89AB_CDEF),
                    )
                    idx = (rand * digit_img[p].shape[1]).long()
                    data[p][q]["img"] = digit_img[p][data[p][q]["num"].long(), idx]

       

        if self.read_satnet_data_path is not None:
            logger.info("Overwriting data from satnet data path: {}".format(self.read_satnet_data_path))
            satdata = torch.load(self.read_satnet_data_path)
            tr,v,t = self.splits['train'], self.splits['val'], self.splits['test']
            
            data['train']['query']['bit'] = satdata['features'][:tr].reshape(tr,-1).float()
            data['train']['query']['img'] = satdata['features_img'][:tr].flatten(start_dim=1,end_dim=2).unsqueeze(2).float()
            data['train']['query']['num'] = utils.zero_one_hot_decode(satdata['features'][:tr].float()).reshape(tr,-1)

            data['train']['target']['bit'] = satdata['labels'][:tr].reshape(tr,-1).float()
            data['train']['target']['img'] = satdata['features_img'][:tr].flatten(start_dim=1,end_dim=2).unsqueeze(2).float()
            data['train']['target']['num'] = utils.zero_one_hot_decode(satdata['labels'][:tr].float()).reshape(tr,-1)

            data['val']['query']['bit'] = satdata['features'][tr:(tr+v)].reshape(v,-1).float()
            data['val']['query']['img'] = satdata['features_img'][tr:(tr+v)].flatten(start_dim=1,end_dim=2).unsqueeze(2).float()
            data['val']['query']['num'] = utils.zero_one_hot_decode(satdata['features'][tr:(tr+v)].float()).reshape(v,-1)
            data['val']['target']['bit'] = satdata['labels'][tr:(tr+v)].reshape(v,-1).float()
            data['val']['target']['img'] = satdata['features_img'][tr:(tr+v)].flatten(start_dim=1,end_dim=2).unsqueeze(2).float()
            data['val']['target']['num'] = utils.zero_one_hot_decode(satdata['labels'][tr:(tr+v)].float()).reshape(v,-1)

            data['test']['query']['bit'] = satdata['features'][-t:].reshape(t,-1).float()
            data['test']['query']['img'] = satdata['features_img'][-t:].flatten(start_dim=1,end_dim=2).unsqueeze(2).float()
            data['test']['query']['num'] = utils.zero_one_hot_decode(satdata['features'][-t:].float()).reshape(t,-1)
            data['test']['target']['bit'] = satdata['labels'][-t:].reshape(t,-1).float()
            data['test']['target']['img'] = satdata['features_img'][-t:].flatten(start_dim=1,end_dim=2).unsqueeze(2).float()
            data['test']['target']['num'] = utils.zero_one_hot_decode(satdata['labels'][-t:].float()).reshape(t,-1)

            for p in ['train','test','val']:
                data[p]['query']['img'] = (data[p]['query']['img'] / 255.0 - 0.1307) / 0.3081
                data[p]['target']['img'] = (data[p]['target']['img'] / 255.0 - 0.1307) / 0.3081



        self.datasets = {}
        for p in ["train", "val", "test"]:
            self.datasets[p] = datasets.DictDataset(utils.flatten_dict(data[p]))
        
        #dump to:
        #Pdb().set_trace()
        if self.write_satnet_data_path is not None:
            dump_to=  '{}_{}tr_{}val_{}test.pt'.format(self.data_path,self.splits['train'],self.splits['val'], self.splits['test'])
       
            dump_dict = {}
            dump_dict['labels'] = torch.cat([data['train']['target']['bit'], 
                                            data['val']['target']['bit'],
                                            data['test']['target']['bit']], dim  = 0).byte().reshape(-1,num_classes, num_classes, num_classes)

            dump_dict['features'] = torch.cat([data['train']['query']['bit'], 
                                            data['val']['query']['bit'],
                                            data['test']['query']['bit']], dim  = 0).byte().reshape(-1,num_classes, num_classes, num_classes)

            ### Originally, the files were written using the commented block.
            ### This is because the images were not normalized by manually commenting the normalization part
            #dump_dict['features_img'] = torch.cat([data['train']['query']['img'], 
            #                                data['val']['query']['img'],
            #                                data['test']['query']['img']], dim  = 0).byte()
            
            dump_dict['features_img'] = torch.cat([data['train']['query']['img'], 
                                            data['val']['query']['img'],
                                            data['test']['query']['img']], dim  = 0).byte()
           
            dump_dict['features_img'] = (255*dump_dict['features_img']*0.3081 + 0.1307).round().byte()

            dump_dict['features_img'] = dump_dict['features_img'].squeeze().reshape(-1,num_classes, num_classes, 28, 28)
            torch.save(dump_dict, dump_to)
            raise


class KnapsackDataModule(datasets.BaseDataModule):
    def __init__(
        self,
        data_path: str,
        splits: dict,
        num_validate_train: int,
        batch_sizes: dict,
        num_workers: int = 0,
    ):
        super().__init__(
            data_path=data_path,
            splits=splits,
            num_validate_train=num_validate_train,
            batch_sizes=batch_sizes,
            num_workers=num_workers,
        )

    def setup(self, stage=None):
        logger.info(f"load {self.data_path}...")
        pt = torch.load(self.data_path, map_location="cpu")
        logger.info("loaded.")

        enc = pt["encodings"]
        rand_idx = pt["rand_idx"]
        y = pt["y"]
        capacity = float(pt["capacity"])

        data = defaultdict(lambda: {})
        for p in ["train", "val", "test"]:

            assert rand_idx[p].shape[0] == y[p].shape[0]
            assert y[p].shape[0] >= self.splits[p]

            data[p]["idx"] = torch.arange(self.splits[p])
            data[p]["x"] = enc[p][rand_idx[p].long()].float()[: self.splits[p]]
            data[p]["y"] = y[p].float()[: self.splits[p]]
            data[p]["cap"] = torch.empty(self.splits[p]).fill_(capacity)

        self.datasets = {}
        for p in ["train", "val", "test"]:
            self.datasets[p] = datasets.DictDataset(data[p])
