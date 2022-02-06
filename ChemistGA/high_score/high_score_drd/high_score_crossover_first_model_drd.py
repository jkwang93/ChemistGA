#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import random

import torch
from rdkit import Chem

from high_score_properties_drd2 import multi_scoring_functions
from transformer_model.onmt.utils.logging import init_logger
from transformer_model.onmt.translate.translator import build_translator

from transformer_model.onmt.opts_translate import OPT_TRANSLATE

import numpy as np


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)

    return ' '.join(tokens)

def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    #mol_from_smarts = Chem.MolFromSmarts(smiles)
    #try:
    #    Chem.SanitizeMol(mol_from_smarts)
    #    mol_from_smarts.UpdatePropertyCache()
    #    return smiles
    #except:
    #    return ''
    return Chem.MolToSmiles(mol, isomericSmiles=True)

def synthesis(opt, src_data_iter):
    torch.cuda.set_device(opt.gpu)
    translator = build_translator(opt, report_score=True)
    # print(src_data_iter)
    all_scores, all_predictions = translator.translate(src_path=opt.src,
                                                       src_data_iter=src_data_iter,
                                                       tgt_path=opt.tgt,
                                                       src_dir=opt.src_dir,
                                                       batch_size=opt.batch_size,
                                                       attn_debug=opt.attn_debug)
    torch.cuda.empty_cache()

    # selecte the highest score from all_predictions of each pair of molecules
    # print(all_predictions)
    return all_predictions


def get_synthesis_molecules(tgt_data_iter):
    opt = OPT_TRANSLATE()
    token_list = []
    for smi in tgt_data_iter:
        token_smi = smi_tokenizer(smi)
        token_list.append(token_smi)
    all_predictions = synthesis(opt, token_list)

    # 去掉所有空格
    # 如果一个反应生成了两个分子，molecular transform会用.连接，先将其分开
    population = []
    all_score = []
    all_active_list = []
    all_population = []
    all_population_score = []
    for pair in all_predictions:
        each_pair_synthesis_list = []
        for each in pair:
            raw_smile = each.replace(" ", "").split('.')
            for sm in raw_smile:
                if Chem.MolFromSmiles(sm) != None:
                    # smi = canonicalize_smiles(sm)
                    each_pair_synthesis_list.append(sm)

            each_pair_synthesis_list = set(each_pair_synthesis_list)
            each_pair_synthesis_list = list(each_pair_synthesis_list)
            # each_pair_synthesis_list.extend(raw_smile)

        # todo 取分数最大的任意一个，最简单的方法就是每个分子取一个，但是这样会多次调用scoring function，很浪费时间，所以想办法只调用一次
        score = multi_scoring_functions(each_pair_synthesis_list, ['drd2', 'qed', 'sa'])
        print('最大得分: ',max(score))
        # print(each_pair_synthesis_list)

        # max_index = np.where(score == max(score))

        # if max(score) == 3:
        #     for ind in max_index[0]:
        #         ac_mol = each_pair_synthesis_list[ind]
        #         all_active_list.append(ac_mol)
        #         # print('satisify: ', ac_mol)

        # 多选几个, 逆序输出他们的index值
        nev_sort_index = np.argsort(-score)
        best_3_index = nev_sort_index[:3]
        # sample_index = random.sample(list(max_index[0]), 1)[0]
        sons = list(np.array(each_pair_synthesis_list)[[best_3_index]])
        sons_scores = list(np.array(score)[[best_3_index]])

        # 记录所有推荐的分子
        all_population.extend(each_pair_synthesis_list)
        all_population_score.extend(score)
        # TODO score也需要取3个
        population.extend(sons)
        # print('population: ',population)
        all_score.extend(sons_scores)

    return all_score, population, all_population,all_population_score

if __name__ == "__main__":
    opt = OPT_TRANSLATE()

    logger = init_logger(opt.log_file)
    # synthesis(opt,tgt_data_iter)
