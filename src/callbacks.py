import pytorch_lightning as pl
from IPython.core.debugger import Pdb 
import time
import json
import os
import logging

logger = logging.getLogger(__name__)

class CustomLogging(pl.Callback):
    def __init__(self,rel_filename='training_stats.json', global_filename='all_stats.json'):
        self.filename = rel_filename
        self.abs_filename = global_filename
        self.logging_dir = None
    
    def log_metric_dict(self,metric_dict, trainer, pl_module, meta_data = {}): 
        self.set_filepath(trainer)
        meta_data.update({k:v for k,v in metric_dict.items()})
        meta_data['log_dir'] = self.logging_dir
        meta_data['test_only'] = pl_module.config.get('test_only',0)
        log_str = json.dumps(meta_data)
        with open(self.filename,'a') as fh:
            print(log_str,file=fh)
        with open(self.abs_filename,'a') as fh:
            print(log_str,file=fh)



    def set_filepath(self,trainer):
        if self.logging_dir is None:
            self.logging_dir = trainer.logger.log_dir
            self.filename = os.path.join(self.logging_dir, self.filename) 
            global_dir = os.path.dirname(self.abs_filename)
            if global_dir != '' and not os.path.isdir(global_dir):
                os.makedirs(global_dir)

    def on_train_epoch_start(self, trainer, pl_module):
        pass

    def on_train_epoch_end(self, trainer, pl_module):
        pass
        
    def on_validation_epoch_start(self, trainer, pl_module):
        pass
    
    def on_validation_epoch_end(self, trainer, pl_module):
        self.log_metric_dict(pl_module.latest_metric_values, trainer,pl_module,
            {'event': 'val_epoch_end','epoch': pl_module.current_epoch})
        
    def on_test_epoch_start(self, trainer, pl_module):
        pass

    def on_test_epoch_end(self, trainer, pl_module):
        self.log_metric_dict(pl_module.latest_metric_values,trainer,pl_module, 
            {'event': 'test_epoch_end','epoch': pl_module.current_epoch})
        

class Timer(pl.Callback):
    def __init__(self):
        self.total_time = {
            "train": 0,
            "val": 0,
            "test": 0,
        }
        self.start_time = {
            "train": None,
            "val": None,
            "test": None,
        }

    def on_train_epoch_start(self, trainer, pl_module):
        logger.info(f"start train epoch {pl_module.current_epoch}")
        self.start_time["train"] = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        logger.info(f"end train epoch {pl_module.current_epoch}")
        epoch_time = time.time() - self.start_time["train"]
        self.total_time["train"] += epoch_time
        train_times = {
                        "epoch_time/train": epoch_time,
                        "total_time/train": self.total_time["train"],
                      } 
        pl_module.log_dict(train_times)
        pl_module.latest_metric_values.update(train_times)


    def on_validation_epoch_start(self, trainer, pl_module):
        logger.info(f"start val epoch {pl_module.current_epoch}")
        self.start_time["val"] = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        logger.info(f"end val epoch {pl_module.current_epoch}")
        epoch_time = time.time() - self.start_time["val"]
        self.total_time["val"] += epoch_time
        val_times = {
                "epoch_time/val": epoch_time,
                "total_time/val": self.total_time["val"],
            }

        pl_module.log_dict(val_times)
        pl_module.latest_metric_values.update(val_times)


    def on_test_epoch_start(self, trainer, pl_module):
        logger.info(f"start test epoch {pl_module.current_epoch}")
        self.start_time["test"] = time.time()

    def on_test_epoch_end(self, trainer, pl_module):
        logger.info(f"end test epoch {pl_module.current_epoch}")
        epoch_time = time.time() - self.start_time["test"]
        self.total_time["test"] += epoch_time
        test_times = {
                "epoch_time/test": epoch_time,
                "total_time/test": self.total_time["test"],
            }
 
        pl_module.log_dict(test_times)
        pl_module.latest_metric_values.update(test_times)
