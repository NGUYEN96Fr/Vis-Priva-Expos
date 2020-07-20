"""
This file support to fine-tune detection models of the detectron2 on customized datasets.
"""
import os
import yaml
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog


def load_custom_train_dataset(params):
    """"""
    register_coco_instances(params['dataset']['train']['name'], {},  params['dataset']['train']['ann_path'],
                            params['dataset']['train']['img_path'])
    metadata = MetadataCatalog.get(params['dataset']['train']['name'])
    dataset_dicts = DatasetCatalog.get(params['dataset']['train']['name'])

    return metadata, dataset_dicts


def config(params):
    """"""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(os.path.join("COCO-Detection", params['model']['name']+".yaml")))
    cfg.DATASETS.TRAIN = (params['dataset']['train']['name'],)
    cfg.DATASETS.TEST = (params['dataset']['train']['name'],)
    cfg.DATALOADER.NUM_WORKERS = params['data_loader']['num_workers']
    cfg.MODEL.WEIGHTS = params['model']['weight']
    cfg.SOLVER.IMS_PER_BATCH = params['solver']['ims_per_batch']
    cfg.SOLVER.BASE_LR = params['solver']['base_lr']  # pick a good LR
    cfg.SOLVER.MAX_ITER = params['solver']['max_iter']
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = params['model']['roi_heads']['batch_size_per_image']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = params['model']['roi_heads']['num_classes']
    cfg.OUTPUT_DIR = params['output_dir']['path']

    return cfg

def main():
    """"""

    params = yaml.safe_load(open('./configs.yaml').read())
    print('Custom Parameters: ')
    print(params)
    metadata, dataset_dicts = load_custom_train_dataset(params)
    cfg = config(params)
    print('Configuration: ')
    print(cfg)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == '__main__':
    main()
