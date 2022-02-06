'''
Jikewang
'''

import pandas as pd
from rdkit import Chem

import numpy as np
import random
import time
import mutate_raw as mu
from scoring.properties_one_hot import multi_scoring_functions_one_hot

from crossover_synthesis_forward import get_synthesis_molecules

import warnings

warnings.filterwarnings("ignore")


def read_file(file_name):
    smiles_list = pd.read_csv(file_name, header=None).values.flatten().tolist()

    return smiles_list


def make_initial_population(population_size, file_name):
    mol_list = read_file(file_name)
    population = []
    for i in range(population_size):
        population.append(random.choice(mol_list))

    return population





# 从beam search的所有反应中，按照scoring挑选一个
# def selected_each_synthesis():


def calculate_normalized_fitness(population, age, score_list=None):
    # Scoring_function
    if age == 0:
        score = multi_scoring_functions_one_hot(population, ['jnk3', 'gsk3', 'qed', 'sa'])
    else:
        score = score_list
    print('age: ',age)

    # max_index = np.where(score == max(score))
    # print(age)
    # if age != 0:
    #     if max(score == 4):
    #         print(max_index)
    #         for ind in max_index[0]:
    #             ac_mol = Chem.MolToSmiles(population[ind])
    #             all_active_list.append(ac_mol)
    #             # print(ac_mol)
    #
    # sample_index = random.sample(list(max_index[0]), 1)[0]
    #
    # global max_score
    #
    # max_score = [max(score), Chem.MolToSmiles(population[sample_index])]

    fitness = score
    # print(fitness)

    # calculate probability
    sum_fitness = sum(fitness)
    normalized_fitness = [fit_idv / sum_fitness for fit_idv in fitness]

    return normalized_fitness


def make_mating_pool(population, fitness):
    mating_pool = []
    # 无重复随机抽取population_size个分子
    for i in range(population_size):
        mating_pool.append(np.random.choice(population, p=fitness, replace=False))

    return mating_pool


def reproduce(mating_pool, population_size, mutation_rate):
    parent_population = []
    new_population = []
    score_list = []
    all_active_list = []
    # count = len(parent_population)
    while len(parent_population) < population_size:
        parent_A = random.choice(mating_pool)
        parent_B = random.choice(mating_pool)
        # print (Chem.MolToSmiles(parent_A),Chem.MolToSmiles(parent_B))
        # new_child = co.crossover(parent_A, parent_B)
        # parent_population.append(parent_A + '.' + parent_B)
        # 去掉重复的population
        # parent_population = set(parent_population)
        # parent_population = list(parent_population)
        # count = len(parent_population)
        # print(count)

        parent_list = [parent_A, parent_B]
        parent_list.sort()
        parent_population.append('.'.join(parent_list))
        # 去掉重复的population
        parent_population = set(parent_population)
        parent_population = list(parent_population)

    # print('选出的population: ',parent_population)

    score_list, new_child, all_active_list = get_synthesis_molecules(parent_population)
    # new_child = set(new_child)
    # print('new_child_len: ', len(new_child))
    # print('score: ', score)
    # print('child: ', new_child)
    for mol in new_child:
        mol = Chem.MolFromSmiles(mol)
        if mol != None:
            mol_new_child = mu.mutate(mol, mutation_rate)
            # print "after mutation",new_child
            if mol_new_child != None:
                # print(Chem.MolToSmiles(new_child))
                new_population.append(Chem.MolToSmiles(mol_new_child))
                # todo 其实应该再算一次，先存着再说吧
                # score_list.append(score)
                # all_active_list.append(all_active)

    # print('new_population: ',new_population)
    return score_list, new_population, all_active_list


global max_score
global count



population_size = 100
generations = 51
mutation_rate = 0.01


print('population_size', population_size)
print('generations', generations)
print('mutation_rate', mutation_rate)
print('')

file_name = './inh/jnk_gsk.csv'

results = []
size = []
t0 = time.time()
all_active_list = []
for i in range(101):

    max_score = [-99999., '']
    count = 0
    population = make_initial_population(population_size, file_name)
    age = 0
    # 初始化score_list = None
    score_list = None
    try:
        for generation in range(generations):
            # 记录第几代
            # if generation%10 == 0: print generation
            # 第一代的不进入活性文件
            fitness = calculate_normalized_fitness(population, age, score_list)
            mating_pool = make_mating_pool(population, fitness)

            score_list, population, active = reproduce(mating_pool, population_size, mutation_rate)

            # 每一步都写
            every_active =  pd.DataFrame(active)
            every_active.to_csv('./output/first_model_no_canonical_large.smi', index=False, header=False, mode='a')

            all_active_list.extend(active)
            # print(all_active_list)

            age += 1
            print('len of set: ', len(set(all_active_list)))
    except Exception as e:
        continue
    print(i, max_score[0], max_score[1], Chem.MolFromSmiles(max_score[1]).GetNumAtoms())
    # print(max_score[1])

    results.append(max_score[0])
    size.append(Chem.MolFromSmiles(max_score[1]).GetNumAtoms())

    print('len of list: ', len(all_active_list))
    print('len of set: ', len(set(all_active_list)))
    if len(set(all_active_list)) > 10000000:
        all_active_list = pd.DataFrame(set(all_active_list))
        all_active_list.to_csv('./output/first_model_no_canonical_large.smi', index=False, header=False)
        break

t1 = time.time()
print('')
print('time ', t1 - t0)
print(max(results), np.array(results).mean(), np.array(results).std())
print(max(size), np.array(size).mean(), np.array(size).std())

# torch.cuda.empty_cache()

