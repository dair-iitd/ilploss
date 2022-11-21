from .CombOptNet.src import models, ilp
import dgl
from .rrn import RRN
from torch import nn
import torch
import torchmetrics as tm
from . import utils
from IPython.core.debugger import Pdb
import seaborn as sns
from torchvision.utils import make_grid
from typing import Optional
import torch.nn.functional as F
import logging
import re

logger = logging.getLogger(__name__)
class SudokuNNModel(models.MyLightningModule):
    def __init__(
        self,
        num_classes: int,
        optimizer: dict,
        lr_scheduler: dict,
        c_encoder: nn.Module,
        network: nn.Module,
        schedulers: dict = {},
        classifier_loss_wt: float = 0,
        classifier_train_key: Optional[str] = None,
        x_key: str = "query/bit",
        y_key: str = "target/bit",
    ):
        super().__init__(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            schedulers=schedulers,
        )
        self.classifier_loss_wt = classifier_loss_wt
        self.num_classes = num_classes
        self.c_encoder = c_encoder
        self.net = network

        self.x_key = x_key
        self.y_key = y_key
        self.classifier_train_key = classifier_train_key
        self.metrics.update(
            {
                "train/acc/idv": tm.Accuracy(subset_accuracy=False),
                "train/acc/all": tm.Accuracy(subset_accuracy=True),
                "val/acc/idv": tm.Accuracy(subset_accuracy=False),
                "val/acc/all": tm.Accuracy(subset_accuracy=True),
                "test/acc/idv": tm.Accuracy(subset_accuracy=False),
                "test/acc/all": tm.Accuracy(subset_accuracy=True),
                "train/run_acc/idv": tm.Accuracy(subset_accuracy=False),
                "train/run_acc/all": tm.Accuracy(subset_accuracy=True),
            }
        )
        self.my_hparams.update(
            {
                "": utils.hparams(self.c_encoder),
            }
        )
        self.my_hparams.update(
            {
                "": utils.hparams(self.net),
            }
        )
        self.hparam_metrics += ["val/acc/idv", "val/acc/all"]

        if x_key == "query/img":
            self.metrics.update(
                {
                    "train/acc/pre": tm.Accuracy(subset_accuracy=False),
                    "val/acc/pre": tm.Accuracy(subset_accuracy=False),
                    "test/acc/pre": tm.Accuracy(subset_accuracy=False),
                    "train/run_acc/pre": tm.Accuracy(subset_accuracy=False),
                },
            )
        if y_key == "target/bit":
            self.metrics.update(
                {
                    "train/acc/num": tm.Accuracy(subset_accuracy=True),
                    "val/acc/num": tm.Accuracy(subset_accuracy=True),
                    "test/acc/num": tm.Accuracy(subset_accuracy=True),
                    "train/run_acc/num": tm.Accuracy(subset_accuracy=True),
                },
            )
        self.my_hparams.update({})
        self.hparam_metrics += ["val/acc/num", "val/status/optimal","test/acc/num", "test/acc/all"]
        self.latest_metric_values = {}


    def reset_val_metrics(self):
        for k, v in self.metrics.items():
            if self.is_val_metric(k):
                v.reset()


    def update_latest_metric_values(self, key_value_pairs, key_prefix='', key_suffix='',filter_fn=None):
        self.latest_metric_values.update(
                {
                    key_prefix+k+key_suffix:(v.item() if torch.is_tensor(v) else v) for 
                    k,v in key_value_pairs.items() if (filter_fn is None or filter_fn(k,v)) 
                }
        )

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if optimizer_idx != 0:
            return None
        i, x, y = batch["idx"], batch[self.x_key], batch[self.y_key]
        c = self.c_encoder(x)
        classifier_loss = 0.0
        if self.classifier_train_key is not None:
            classifier_loss = F.nll_loss(torch.log(c.transpose(1,2)),  batch[self.classifier_train_key].long()).mean()
        
        pred, loss = self.net(c, batch)
        yhat = utils.zero_one_hot_encode(pred[-1], num_classes = self.num_classes).view(y.shape[0], -1)  
        self.train_step_log(batch, batch_idx,  c, yhat)
        return (1.0-self.classifier_loss_wt)*loss + self.classifier_loss_wt*classifier_loss
    
    def training_step_end(self, loss):
        if loss is not None:
            self.log("train/loss", loss)
        g = [p.grad.view(-1) for p in self.parameters() if p.grad is not None]
        if g:
            self.log("train/grad", torch.linalg.vector_norm(torch.cat(g)))

    def training_epoch_end(self,outputs):
        self.log_dict({k: v for k, v in self.metrics.items() if k.startswith("train/run")})
        if self.y_key == "target/bit":
            logger.info("train/run_acc/num ")
            logger.info(
                " \t"
                + " | ".join(
                    [
                        f"run_train: {self.metrics['train/run_acc/num'].compute().item():.4f}",
                    ]
                )
            )
        logger.info("Running Train:")
        logger.info(
            " \t"
            + " | ".join(
                [
                    f"idv: {self.metrics['train/run_acc/idv'].compute().item():.4f}",
                    f"all: {self.metrics['train/run_acc/all'].compute().item():.4f}",
                ]
            )
        )


    def test_step(self, batch, batch_idx, dataloader_idx=None):
        i, x, y = batch["idx"], batch[self.x_key], batch[self.y_key]
        if i.numel() == 0:
            return torch.tensor([]).cuda()
        # 
        c = self.c_encoder(x)
        pred, loss = self.net(c, batch, is_training=False)
        yhat = utils.zero_one_hot_encode(pred, num_classes = self.num_classes).view(y.shape[0], -1)  

        if batch_idx == 0 and (dataloader_idx is None or dataloader_idx == 0):
            self.test_save = batch, c, yhat

        self.test_step_log(batch, batch_idx, dataloader_idx, c, yhat)
        #Pdb().set_trace()
        return torch.zeros_like(i)

    def on_validation_start(self):
        return 

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        if not self.should_validate():
            return
        # 
        i, x, y = batch["idx"], batch[self.x_key], batch[self.y_key]
        if i.numel() == 0:
            return torch.tensor([]).cuda()
        c = self.c_encoder(x)
        pred, loss = self.net(c, batch,is_training = False)
        yhat = utils.zero_one_hot_encode(pred, num_classes = self.num_classes).view(y.shape[0], -1)  

        if batch_idx == 0 and dataloader_idx == 1:
            self.val_save = batch,  c, yhat

        self.validation_step_log(batch, batch_idx, dataloader_idx,  c, yhat)
        return torch.zeros_like(i)


    def train_step_log(self, batch, batch_idx,  c, yhat):
        r = 'train'
        if self.x_key == "query/img":
            self.metrics[f"{r}/run_acc/pre"](
                torch.argmax(c.contiguous().view(-1, c.shape[-1]), dim=-1)[
                    batch["query/num"].long().view(-1) != 0
                ],
                batch["query/num"].long().view(-1)[batch["query/num"].view(-1) != 0],
            )
        lb = torch.minimum(torch.min(yhat).long(), batch[self.y_key].long())
        self.metrics[f"{r}/run_acc/idv"](yhat.long() - lb, batch[self.y_key].long() - lb)
        self.metrics[f"{r}/run_acc/all"](yhat.long() - lb, batch[self.y_key].long() - lb)

        if self.y_key == "target/bit":
            self.metrics[f"{r}/run_acc/num"](
                yhat.long().view(-1, self.num_classes),
                batch["target/bit"].long().view(-1, self.num_classes),
            )

    def test_step_log(self, batch, batch_idx, dataloader_idx, c, yhat):
        if self.x_key == "query/img":
            self.metrics[f"test/acc/pre"](
                torch.argmax(c.contiguous().view(-1, c.shape[-1]), dim=-1)[
                    batch["query/num"].long().view(-1) != 0
                ],
                batch["query/num"].long().view(-1)[batch["query/num"].view(-1) != 0],
            )

        lb = torch.minimum(torch.min(yhat).long(), batch[self.y_key].long())
        self.metrics[f"test/acc/idv"](yhat.long() - lb, batch[self.y_key].long() - lb)
        self.metrics[f"test/acc/all"](yhat.long() - lb, batch[self.y_key].long() - lb)

        if self.y_key == "target/bit":
            self.metrics[f"test/acc/num"](
                yhat.long().view(-1, self.num_classes),
                batch["target/bit"].long().view(-1, self.num_classes),
            )
    
    def reset_test_metrics(self):
        for k, v in self.metrics.items():
            if k.startswith("test"):
                v.reset()
    
    def validation_step_log(self, batch, batch_idx, dataloader_idx, c, yhat):
        r = "train" if dataloader_idx == 0 else "val"
        if self.x_key == "query/img":
            self.metrics[f"{r}/acc/pre"](
                torch.argmax(c.contiguous().view(-1, c.shape[-1]), dim=-1)[
                    batch["query/num"].long().view(-1) != 0
                ],
                batch["query/num"].long().view(-1)[batch["query/num"].view(-1) != 0],
            )
        lb = torch.minimum(torch.min(yhat).long(), batch[self.y_key].long())
        self.metrics[f"{r}/acc/idv"](yhat.long() - lb, batch[self.y_key].long() - lb)
        self.metrics[f"{r}/acc/all"](yhat.long() - lb, batch[self.y_key].long() - lb)

        if self.y_key == "target/bit":
            self.metrics[f"{r}/acc/num"](
                yhat.long().view(-1, self.num_classes),
                batch["target/bit"].long().view(-1, self.num_classes),
            )

    def is_val_metric(self, x):
        return not (x.startswith('test') or x.startswith('train/run'))

    def validation_epoch_end(self, outputs):
        if not self.should_validate():
            logger.info(f"skip validation for {self.current_epoch}")
            val_metric_values = {k: 0.0 for k, v in self.metrics.items() if self.is_val_metric(k)}
            self.log_dict(val_metric_values)
            self.update_latest_metric_values(val_metric_values)
            self.update_latest_metric_values({'val_update_epoch': self.current_epoch})
            self.reset_val_metrics()
            return val_metric_values
        
        batch, c, yhat = self.val_save
        x, y = batch[self.x_key], batch[self.y_key]

        tb = self.logger.experiment
        sample_id = torch.randint(0,y.shape[0],(1,))[0]
        tb.add_text(
            "sample",
            utils.wrap_text(
                f'yhat: {" ".join(map(str, yhat[sample_id].long().tolist()))}\n'
                f'y   : {" ".join(map(str, y[sample_id].long().tolist()))}'
            ),
            self.global_step,
        )

        
        val_metric_values = {k: v.compute() for k, v in self.metrics.items() if self.is_val_metric(k)}
        self.log_dict(val_metric_values)
        logger.info(f"")
        logger.info(f"EPOCH {self.current_epoch}")
        logger.info("train")
        logger.info(
            " \t"
            + " | ".join(
                [
                    f"idv: {self.metrics['train/acc/idv'].compute().item():.4f}",
                    f"all: {self.metrics['train/acc/all'].compute().item():.4f}",
                ]
            )
        )
        logger.info("val")
        logger.info(
            " \t"
            + " | ".join(
                [
                    f"idv: {self.metrics['val/acc/idv'].compute().item():.4f}",
                    f"all: {self.metrics['val/acc/all'].compute().item():.4f}",
                ]
            )
        )

        if self.x_key == "query/img":
            tb.add_figure(
                "query/img",
                sns.heatmap(
                    make_grid(batch["query/img"][0], nrow=self.num_classes)[0]
                    .detach()
                    .cpu(),
                    cmap="gray",
                ).get_figure(),
                self.global_step,
            )

        if self.y_key == "target/bit":
            logger.info("acc/num ")
            logger.info(
                " \t"
                + " | ".join(
                    [
                        f"train: {self.metrics['train/acc/num'].compute().item():.4f}",
                        f"val: {self.metrics['val/acc/num'].compute().item():.4f}",
                    ]
                )
            )
        self.reset_val_metrics()
        self.update_latest_metric_values(val_metric_values)
        self.update_latest_metric_values({'val_update_epoch': self.current_epoch})
        return val_metric_values

    def test_epoch_end(self, outputs):
        batch,  c, yhat = self.test_save
        x, y = batch[self.x_key], batch[self.y_key]

        tb = self.logger.experiment
        if self.x_key == "query/img":
            tb.add_figure(
                "query/img",
                sns.heatmap(
                    make_grid(batch["query/img"][0], nrow=self.num_classes)[0]
                    .detach()
                    .cpu(),
                    cmap="gray",
                ).get_figure(),
                self.global_step,
            )
        tb.add_text(
            "sample",
            utils.wrap_text(
                f"yhat: {' '.join(map(str, yhat[0].long().tolist()))}\n"
                f"y   : {' '.join(map(str, y[0].long().tolist()))}"
            ),
            self.global_step,
        )

        test_metrics = {
            k: v.compute().item() for k, v in self.metrics.items() if k.startswith("test")
        }
        self.reset_test_metrics()
            
        self.log_dict(test_metrics)
        #self.log_dict({k: v for k, v in self.metrics.items() if k.startswith("test")})

        if self.y_key == "target/bit":
            logger.info("acc/num ")
            logger.info(
                " \t"
                + " | ".join(
                    [
                        f"test: {test_metrics['test/acc/num']:.4f}",
                    ]
                )
            )

        self.update_latest_metric_values(test_metrics)
        self.update_latest_metric_values({'test_update_epoch': self.current_epoch})
        logger.info("test")
        logger.info(
            " \t"
            + " | ".join(
                [
                    f"idv: {test_metrics['test/acc/idv']:.4f}",
                    f"all: {test_metrics['test/acc/all']:.4f}",
                ]
            )
        )

"""
SudokuNN module based on RRN for solving sudoku puzzles
"""



class SudokuNN(nn.Module):
    def __init__(self,
                 num_steps,
                 embed_size=16,
                 hidden_dim=96,
                 edge_drop=0.1,
                 num_classes=9,
                ):
        super(SudokuNN, self).__init__()
        self.num_steps = num_steps
        self.board_size = num_classes 
        self.digit_embed = nn.Embedding(self.board_size+1, embed_size)
        self.row_embed = nn.Embedding(self.board_size, embed_size)
        self.col_embed = nn.Embedding(self.board_size, embed_size)

        self.input_layer = nn.Sequential(
            nn.Linear(3*embed_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.lstm = nn.LSTMCell(hidden_dim*2, hidden_dim, bias=False)

        msg_layer = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.rrn = RRN(msg_layer, self.node_update_func, num_steps, edge_drop)

        self.output_layer = nn.Linear(hidden_dim, self.board_size+1)

        self.loss_func = nn.CrossEntropyLoss()
        self.hparams = {"num_steps": self.num_steps }


    def forward(self, cost, batch, is_training=True):
        g = batch['batch_graph']
        labels = g.ndata.pop('a')
        input_digits = cost.view(cost.shape[0]*cost.shape[1],cost.shape[2]) @ self.digit_embed.weight
        #input_digits = self.digit_embed(g.ndata.pop('q'))
        rows = self.row_embed(g.ndata.pop('row'))
        cols = self.col_embed(g.ndata.pop('col'))
        x = self.input_layer(torch.cat([input_digits, rows, cols], -1))
        g.ndata['x'] = x
        g.ndata['h'] = x
        g.ndata['rnn_h'] = torch.zeros_like(x, dtype=torch.float)
        g.ndata['rnn_c'] = torch.zeros_like(x, dtype=torch.float)

        outputs = self.rrn(g, is_training)
        logits = self.output_layer(outputs)

        preds = torch.argmax(logits, -1)

        if is_training:
            labels = torch.stack([labels]*self.num_steps, 0)
        logits = logits.view([-1, self.board_size+1])
        labels = labels.view([-1])
        loss = self.loss_func(logits, labels.long())
        return preds, loss

    def node_update_func(self, nodes):
        x, h, m, c = nodes.data['x'], nodes.data['rnn_h'], nodes.data['m'], nodes.data['rnn_c']
        new_h, new_c = self.lstm(torch.cat([x, m], -1), (h, c))
        return {'h': new_h, 'rnn_c': new_c, 'rnn_h': new_h}


