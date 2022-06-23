# Implementation adapted from XNAS: https://github.com/MAC-AutoML/XNAS

from core.config import cfg
from datasets.imagenet import ImageFolder


# if you use datasets loaded by imagefolder, you can add it here.
IMAGEFOLDER_FORMAT = ["imagenet"]


def get_normal_dataloader(
    name=None,
    train_batch=None,
    **kwargs
):
    name=cfg.LOADER.DATASET if name is None else name
    train_batch=cfg.LOADER.BATCH_SIZE if train_batch is None else train_batch
    name=cfg.LOADER.DATASET
    datapath=cfg.LOADER.DATAPATH
    test_batch=cfg.LOADER.BATCH_SIZE if cfg.TEST.BATCH_SIZE == -1 else cfg.TEST.BATCH_SIZE
    
    assert (name in IMAGEFOLDER_FORMAT), "dataset not supported."
    assert isinstance(train_batch, int), "normal dataloader using single training batch-size, not list."
    # check if randomresized crop is used only in ImageFolder type datasets
    # if len(cfg.SEARCH.MULTI_SIZES):
    #     assert name in IMAGEFOLDER_FORMAT, "RandomResizedCrop can only be used in ImageFolder currently."

    assert cfg.LOADER.USE_VAL is True, "getting normal dataloader."
    aug_type = cfg.LOADER.TRANSFORM
    return ImageFolder( # using path of training data of ImageNet as `datapath`
        datapath, batch_size=[train_batch, test_batch],
        use_val=True, augment_type=aug_type, **kwargs
    ).generate_data_loader()
