"""
This file support to fine-tune detection models of the detectron2 on customized datasets.

Ref codes: https://github.com/facebookresearch/detectron2/blob/master/projects/TridentNet/train_net.py
            https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/defaults.py

USE:
    training:
    ---------
    python retinanet_custom.py --config_file configs/retinanet.yaml --num_gpus 4

    test:
    -----
    python retinanet_custom.py --config_file configs/retinanet.yaml --eval-only

"""
import os

import  detectron2
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


def load_custom_train_dataset():
    """"""
    train_name = 'train_v0_Retinanet'
    train_img_path = '/home/users/vnguyen/intern20/DATA/CUSTOM/V0/train'
    train_ann_path = '/home/users/vnguyen/intern20/DATA/CUSTOM/V0/annotations/instances_train.json'

    register_coco_instances(train_name, {}, train_ann_path,
                            train_img_path)
    metadata = MetadataCatalog.get(train_name)
    dataset_dicts = DatasetCatalog.get(train_name)

    return metadata, dataset_dicts


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    load_custom_train_dataset()
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
