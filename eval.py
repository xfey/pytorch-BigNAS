# Implementation adapted from XNAS: https://github.com/MAC-AutoML/XNAS

"""BigNAS subnet evaluation"""

import torch

import core.config as config
import logger.meter as meter
import logger.logging as logging
from core.builder import setup_env
from core.config import cfg
from datasets.loader import get_normal_dataloader
from logger.meter import TestMeter

from bignas.cnn import _infer_BigNAS_CNN


# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)


def main():
    setup_env()
    net = _infer_BigNAS_CNN()

    [train_loader, valid_loader] = get_normal_dataloader()
    test_meter = TestMeter(len(valid_loader))

    # Validate
    top1_err, top5_err = validate(net, train_loader, valid_loader, test_meter)
    logger.info("top1_err:{} top5_err:{}".format(top1_err, top5_err))


@torch.no_grad()
def validate(subnet, train_loader, valid_loader, test_meter):
    # BN calibration
    subnet.eval()
    logger.info("Calibrating BN running statistics.")
    subnet.reset_running_stats_for_calibration()
    for cur_iter, (inputs, _) in enumerate(train_loader):
        if cur_iter >= cfg.BIGNAS.POST_BN_CALIBRATION_BATCH_NUM:
            break
        inputs = inputs.cuda()
        subnet(inputs)      # forward only

    top1_err, top5_err = test_epoch(subnet, valid_loader, test_meter)
    return top1_err, top5_err


def test_epoch(subnet, test_loader, test_meter):
    subnet.eval()
    test_meter.reset(True)
    test_meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        preds = subnet(inputs)
        top1_err, top5_err = meter.topk_errors(preds, labels, [1, 5])
        top1_err, top5_err = top1_err.item(), top5_err.item()

        test_meter.iter_toc()
        test_meter.update_stats(top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
        test_meter.log_iter_stats(0, cur_iter)
        test_meter.iter_tic()
    top1_err = test_meter.mb_top1_err.get_win_avg()
    top5_err = test_meter.mb_top5_err.get_win_avg()
    # self.writer.add_scalar('val/top1_error', test_meter.mb_top1_err.get_win_avg(), cur_epoch)
    # self.writer.add_scalar('val/top5_error', test_meter.mb_top5_err.get_win_avg(), cur_epoch)
    # Log epoch stats
    test_meter.log_epoch_stats(0)
    # test_meter.reset()
    return top1_err, top5_err


if __name__ == "__main__":
    main()