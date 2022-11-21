import pytorch_lightning as pl
import sys

# sys.path.insert(0,'src/GraphMatching')
from src.GraphMatching.dataset.data_loader_multigraph import GMDataset, get_dataloader
from src.GraphMatching.utils.config import cfg

from .CombOptNet.src import datasets
from src.GraphMatching.utils.utils import update_params_from_cmdline
from IPython.core.debugger import Pdb


class GMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg_path,
        num_keypoints=0,
        eval_batch_size=8,
        train_batch_size=8,
        train_epoch_iters=400,
        eval_samples=16,
        test_samples=16,
        train_eval_samples=0,
        num_workers=0,
        eval_on_test=False,
    ):
        super().__init__()
        global cfg
        cfg = update_params_from_cmdline(cmd_line=["", cfg_path], default_params=cfg)
        cfg.EVAL.SAMPLES = eval_samples
        if cfg.EVAL.SAMPLES == "None":
            cfg.EVAL.SAMPLES = None

        train_eval_samples = train_eval_samples if train_eval_samples > 0 else None

        self.eval_on_test = eval_on_test
        self.eval_batch_size = eval_batch_size
        self.train_batch_size = train_batch_size
        self.num_workers = num_workers
        # dataset_len = {"train": cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, "test": cfg.EVAL.SAMPLES}
        dataset_len = {
            "train": train_epoch_iters * train_batch_size,
            "test": test_samples if test_samples > 0 else None,
            "val": eval_samples if eval_samples > 0 else None,
        }
        self.sets = ["train", "test", "val"]
        self.image_dataset = {
            x: GMDataset(
                cfg.DATASET_NAME,
                sets=x,
                length=dataset_len[x],
                obj_resize=(256, 256),
                num_keypoints=num_keypoints,
            )
            for x in self.sets
        }
        self.train_as_test = GMDataset(
            cfg.DATASET_NAME,
            sets="train",
            length=train_eval_samples,
            obj_resize=(256, 256),
            num_keypoints=num_keypoints,
        )

    def train_dataloader(self):
        x = "train"
        train_dataloader = get_dataloader(
            self.image_dataset[x],
            num_workers=self.num_workers,
            batch_size=self.train_batch_size,
            fix_seed=False,
        )
        train_dataloader.dataset.set_num_graphs(
            cfg.TRAIN.num_graphs_in_matching_instance
        )
        return train_dataloader

    def val_dataloader(self):
        x = "test" if self.eval_on_test else "val"
        val_dataloader = get_dataloader(
            self.image_dataset[x],
            num_workers=self.num_workers,
            batch_size=self.eval_batch_size,
            fix_seed=True,
        )
        val_dataloader.dataset.set_num_graphs(cfg.EVAL.num_graphs_in_matching_instance)
        train_subset_dataloader = get_dataloader(
            self.train_as_test,
            num_workers=self.num_workers,
            batch_size=self.eval_batch_size,
            fix_seed=True,
        )
        train_subset_dataloader.dataset.set_num_graphs(
            cfg.EVAL.num_graphs_in_matching_instance
        )
        return [train_subset_dataloader, val_dataloader]

    def test_dataloader(self):
        x = "test"
        val_dataloader = get_dataloader(
            self.image_dataset[x],
            num_workers=self.num_workers,
            batch_size=self.eval_batch_size,
            fix_seed=True,
        )
        val_dataloader.dataset.set_num_graphs(cfg.EVAL.num_graphs_in_matching_instance)
        return val_dataloader
