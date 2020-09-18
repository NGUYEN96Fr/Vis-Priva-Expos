import _init_paths
from vispel.config import get_cfg

cfg = get_cfg()

config_file = '../configs/BASE-VISPEL.yaml'
cfg.merge_from_file(config_file)
print(type(cfg.MODEL.RF.BOOTSTRAP))
