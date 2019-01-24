# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import re
import sys
from logging import Handler
from tensorboardX import SummaryWriter


def setup_logger(name, save_dir, distributed_rank):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(TensorboardHandler(save_dir))

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class TensorboardHandler(Handler):

    def __init__(self, save_dir):
        Handler.__init__(self)
        self.log_dir = save_dir
        self.summary_writer = SummaryWriter(log_dir=self.log_dir)
        self.last_msg = ''

        p_loss = ['iter', 'loss', 'loss_box_reg', 'loss_classifier', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg',
                  'lr', 'max mem']
        p_num = '(\d*\.*\d*)'
        p_split = '.+?'

        self.patterns = []
        for p in p_loss:
            pattern_text = p + ':' + p_split + p_num
            self.patterns.append([p, re.compile(pattern_text)])

        self.save_pattern = 'Saving checkpoint to'

    def emit(self, record):
        try:
            msg = record.msg

            if self.save_pattern in msg:
                values = {}
                for label, pattern in self.patterns:
                    m = pattern.search(self.last_msg)
                    if m:
                        values[label] = float(m.group(1))
                    else:
                        values[label] = float(0)

                for label, value in values.items():
                    if label == 'iter':
                        continue
                    self.summary_writer.add_scalar(label, value, global_step=int(values['iter']))

        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

        self.last_msg = msg
