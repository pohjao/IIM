from trainer import Trainer
from config import cfg
from importlib import import_module
import os
import shutil

data_mode = cfg.DATASET
datasetting = import_module(f'datasets.setting.{data_mode}')
cfg_data = datasetting.cfg_data
pwd = os.path.split(os.path.realpath(__file__))[0]
if os.path.isdir("exp/DEBUG"):
    shutil.rmtree("exp/DEBUG", ignore_errors=False, onerror=None)
trainer = Trainer(cfg_data, pwd)
trainer.validate()
