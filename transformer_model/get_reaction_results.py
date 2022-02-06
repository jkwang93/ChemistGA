#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import torch

from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator

from onmt.opts_translate import OPT_TRANSLATE


def synthesis(opt, tgt_data_iter):
    torch.cuda.set_device(opt.gpu)
    translator = build_translator(opt, report_score=True)
    all_scores, all_predictions = translator.translate(src_path=opt.src,
                                                       tgt_data_iter=tgt_data_iter,
                                                       tgt_path=opt.tgt,
                                                       src_dir=opt.src_dir,
                                                       batch_size=opt.batch_size,
                                                       attn_debug=opt.attn_debug)

    return all_predictions







if __name__ == "__main__":
    opt = OPT_TRANSLATE()

    logger = init_logger(opt.log_file)
    # synthesis(opt,tgt_data_iter)
