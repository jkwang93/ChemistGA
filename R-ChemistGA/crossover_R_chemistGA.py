#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import random

import torch
from rdkit import Chem

from scoring.properties_one_hot import multi_scoring_functions_one_hot
from scoring.properties_one_hot_bad import multi_scoring_functions_one_hot_bad
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


def get_synthesis_molecules(tgt_data_iter,age):
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
    all_active_score_list = []
    for pair in all_predictions:
        each_pair_synthesis_list = []
        for each in pair:
            # split .
            raw_smile = each.replace(" ", "").split('.')
            # to canonical smiles
            for sm in raw_smile:
                if Chem.MolFromSmiles(sm)!=None:
                    #smi = canonicalize_smiles(sm)
                    each_pair_synthesis_list.append(sm)

        each_pair_synthesis_list = set(each_pair_synthesis_list)
        each_pair_synthesis_list = list(each_pair_synthesis_list)

        # print(each_pair_synthesis_list)

        # todo 取分数最大的任意一个，最简单的方法就是每个分子取一个，但是这样会多次调用scoring function，很浪费时间，所以想办法只调用一次
        if age%5==0:        
           score = multi_scoring_functions_one_hot(each_pair_synthesis_list, ['jnk3', 'gsk3', 'qed', 'sa'])
         
           print('最大得分: ',max(score))
        # print(each_pair_synthesis_list)

           max_index = np.where(score == max(score))

           if max(score) == 4:
               for ind in max_index[0]:
                   ac_mol = each_pair_synthesis_list[ind]
                   all_active_list.append(ac_mol)
                   all_active_score_list.append(score[ind])
                   # print('satisify: ', ac_mol)
        else:
           score = multi_scoring_functions_one_hot_bad(each_pair_synthesis_list, ['jnk3', 'gsk3', 'qed', 'sa'])
        # 多选几个, 逆序输出他们的index值
        nev_sort_index = np.argsort(-score)
        best_3_index = nev_sort_index[:3]
        # sample_index = random.sample(list(max_index[0]), 1)[0]
        sons = list(np.array(each_pair_synthesis_list)[[best_3_index]])
        sons_scores = list(np.array(score)[[best_3_index]])


        # TODO score也需要取3个
        population.extend(sons)
        # print('population: ',population)
        all_score.extend(sons_scores)

    return all_score, population, all_active_list,all_active_score_list

if __name__ == "__main__":
    opt = OPT_TRANSLATE()

    logger = init_logger(opt.log_file)
    # synthesis(opt,tgt_data_iter)
