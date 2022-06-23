import os
import torch
import random
import numpy as np

import core.config as config
# Implementation adapted from XNAS: https://github.com/MAC-AutoML/XNAS

import logger.logging as logging
from core.config import cfg


logger = logging.get_logger(__name__)


def setup_env():
    """Set up environment for training or testing."""
    # Ensure the output dir exists and save config
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    config.dump_cfgfile()

    # Setup logging
    logging.setup_logging()
    # Log the config as both human readable and as a json
    logger.info("Config:\n{}".format(cfg))
    logger.info(logging.dump_log_data(cfg, "cfg"))
    if cfg.DETERMINSTIC:
        # Fix RNG seeds
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)
        random.seed(cfg.RNG_SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    else:
        # Configure the CUDNN backend
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCH
    device = 'cuda:0'   # TODO: ddp support
    return device
