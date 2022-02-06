from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import os
import pickle
import multiprocessing

from rdchiral.main import rdchiralRunText, rdchiralRun, rdchiralReactants

similarity_metric = DataStructs.BulkTanimotoSimilarity

getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=True)


def load_fps(fps_path):
    with open(fps_path, 'rb') as r:
        fps = pickle.load(r)
    return fps


def get_train_data(use_data, fps_path_list):
    prod_fps_path, react_fps_path = fps_path_list
    datasub = pd.read_csv(use_data)
    print('Init similarity ...')
    print('Calculating products fingerprints ...')
    all_fps = []
    if not os.path.exists(prod_fps_path) and prod_fps_path is not None:
        prod_smiles = datasub['prod_smiles'].tolist()
        for smi in tqdm(prod_smiles):
            all_fps.append(getfp(smi))
        print('Saving products fingerprints ...')
        with open(prod_fps_path, 'wb') as f:
            pickle.dump(all_fps, f)
        del all_fps
    if not os.path.exists(react_fps_path) and react_fps_path is not None:
        print('Calculating reactants fingerprints ...')
        rcts_ref_fps = []
        rxn_smiles = datasub['rxn_smiles'].tolist()
        for rxn in tqdm(rxn_smiles):
            rcts_ref_fps.append(getfp(rxn.split('>')[0]))
        print('Saving reactants fingerprints ...')
        with open(react_fps_path, 'wb') as f:
            pickle.dump(rcts_ref_fps, f)
        del rcts_ref_fps

    # data_train['prod_fp'] = all_fps
    # data_train['rcts_fp'] = rcts_ref_fps
    print('Init Done.')
    return datasub


def retro_sim(prod, datasub, size=10):
    fp = getfp(prod)
    prod_fp = datasub['prod_fp'].tolist()
    begain = time.time()

    sims = similarity_metric(fp, prod_fp)
    end = time.time()
    print('It took {:.2f}s to calculate similarity.'.format(end - begain))
    js = np.argsort(sims)[::-1]
    probs = {}
    for j in js[:25]:
        jx = datasub.index[j]
        tp = datasub['retro_templates'][jx]
        if tp is np.nan:
            continue
        template = '(' + tp.replace('>>', ')>>')
        rcts_fp = datasub['rcts_fp'][jx]
        try:
            outcomes = rdchiralRunText(template, prod)
        except Exception as e:
            outcomes = []

        for precursors in outcomes:
            precursors_fp = getfp(precursors)
            precursors_sim = similarity_metric(precursors_fp, [rcts_fp])[0]
            if precursors in probs:
                probs[precursors] = max(probs[precursors], precursors_sim * sims[j])
            else:
                probs[precursors] = precursors_sim * sims[j]

    testlimit = size
    smiles_all = []
    score = []
    for r, (prec, prob) in enumerate(sorted(list(probs.items()), key=lambda x: x[1], reverse=True)[:testlimit]):
        smiles_all.append(prec)
        score.append(prob)
    smiles_all_split = [x.split('.') for x in smiles_all]
    return smiles_all_split, score


if __name__ == '__main__':
    train_data_path = '../clean_and_extract_uspto_all/dropbox/cooked_uspto_all/uspto_all.csv'
    run_flag = False
    prod_fps_path = 'prod_fps.pkl'
    react_fps_path = 'react_fps.pkl'
    fps_path_list = [prod_fps_path, react_fps_path]
    load_flag = [os.path.exists(path) for path in [prod_fps_path, react_fps_path]]
    load_list = []
    for i, flag in enumerate(load_flag):
        if flag:
            load_list.append(load_fps(fps_path_list[i]))
            fps_path_list[i] = None
        else:
            load_list.append(None)
            run_flag = True

    if run_flag:
        data_train = get_train_data(train_data_path, fps_path_list)
        prod_fps, rcts_fps = [load_fps(path) for path in fps_path_list]
    else:
        data_train = pd.read_csv(train_data_path)
        prod_fps, rcts_fps = load_list
    print('Merage data ...')
    data_train['prod_fp'] = prod_fps
    data_train['rcts_fp'] = rcts_fps
    begain = time.time()
    outcome, score = retro_sim(
        'Cc1nc(-c2ccc(C=O)cc2)sc1COc1ccc([C@H](CC(=O)N2C(=O)OC[C@@H]2Cc2ccccc2)c2ccon2)cc1',
        data_train)
    end = time.time()
    print(outcome)
    print(score)
    print(len(outcome))
    print('predict used {:.2f}s'.format(end - begain))
