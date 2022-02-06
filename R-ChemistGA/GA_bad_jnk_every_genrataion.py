import pandas as pd
from rdkit import Chem

import numpy as np
import random
import time
import mutate as mu
from scoring.properties_one_hot import multi_scoring_functions_one_hot

from crossover_R_chemistGA import get_synthesis_molecules

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
    print('age: ', age)

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


def reproduce(mating_pool, population_size, mutation_rate,mutate2, age):
    # parent_population = []
    # new_population = []
    # # score_list = []
    # # all_active_list = []
    # # count = len(parent_population)
    #
    #
    # while len(parent_population) < population_size:
    #     parent_A = random.choice(mating_pool)
    #     parent_B = random.choice(mating_pool)
    #     # print (Chem.MolToSmiles(parent_A),Chem.MolToSmiles(parent_B))
    #     # new_child = co.crossover(parent_A, parent_B)
    #     parent_list = [parent_A, parent_B]
    #     parent_list.sort()
    #     parent_population.append('.'.join(parent_list))
    #     # 去掉重复的population
    #     parent_population = set(parent_population)
    #     parent_population = list(parent_population)
    #     # count = len(parent_population)
    #     # print(count)

    parent_population = []
    new_population = []
    score_list = []
    all_active_list = []
    # count = len(parent_population)
    # mutate 2
    # 设置一下
    mutate2_polulation = []

    if age % 10 == 0 and age > 1:
        # if age > 6 and len(all_active_list) > 50:
        mutate2_polulation = mutate2

    while len(parent_population) < population_size:
        parent_A = random.choice(mating_pool)
        if mutate2_polulation:
            parent_B = random.choice(mutate2_polulation)
        else:
            parent_B = random.choice(mating_pool)  # print (Chem.MolToSmiles(parent_A),Chem.MolToSmiles(parent_B))
        # new_child = co.crossover(parent_A, parent_B)
        # parent_population.append(parent_A + '.' + parent_B)
        parent_list = [parent_A, parent_B]
        parent_list.sort()
        parent_population.append('.'.join(parent_list))
        # 去掉重复的population
        parent_population = set(parent_population)
        parent_population = list(parent_population)

    # print('选出的population: ',parent_population)
    score_list, new_child, all_active_list,all_active_score_list = get_synthesis_molecules(parent_population,age)
    # new_child = set(new_child)
    # print('new_child_len: ', len(new_child))
    # print('score: ', score)
    # print('child: ', new_child)
    # drop_score_index = []

    # mutate
    drop_score_index = []
    for index, mol in enumerate(new_child):
        mol = Chem.MolFromSmiles(mol)
        if mol != None:
            mol_new_child = mu.mutate(mol, mutation_rate)
            # print "after mutation",new_child
            # 这个地方还有drop掉它的fitness
            if mol_new_child != None:
                # print(Chem.MolToSmiles(new_child))
                new_population.append(Chem.MolToSmiles(mol_new_child))
                # todo 其实应该再算一次，先存着再说吧
                # score_list.append(score)
                # all_active_list.append(all_active)
            else:
                drop_score_index.append(index)
        else:
            drop_score_index.append(index)
    print('要删除的index: ', drop_score_index)

    # 删除对应的score_list
    score_list = np.array(score_list)
    score_list = np.delete(score_list, drop_score_index, axis=0).tolist()



    # print('new_population: ',new_population)

    return score_list, new_population, all_active_list, all_active_score_list


global max_score
global count

# logP_values = np.loadtxt('logP_values.txt')
# SA_scores = np.loadtxt('SA_scores.txt')
# cycle_scores = np.loadtxt('cycle_scores.txt')
# SA_mean = np.mean(SA_scores)
# SA_std = np.std(SA_scores)
# logP_mean = np.mean(logP_values)
# logP_std = np.std(logP_values)
# cycle_mean = np.mean(cycle_scores)
# cycle_std = np.std(cycle_scores)

population_size = 100
generations = 501
mutation_rate = 0.01

# co.average_size = 39.15
# co.size_stdev = 3.50

print('population_size', population_size)
print('generations', generations)
print('mutation_rate', mutation_rate)
# print('average_size/size_stdev', co.average_size, co.size_stdev)
print('')

file_name = './inh/jnk_gsk.csv'
mutate2 = pd.read_csv('./inh/jnk_gsk.csv').values.flatten().tolist()


results = []
size = []
t0 = time.time()
all_active_list = []
for i in range(20):

    max_score = [-99999., '']
    count = 0
    population = make_initial_population(population_size, file_name)
    age = 0
    # 初始化score_list = None
    score_list = None

    for generation in range(generations):
        print(age)
        # 记录第几代
        # if generation%10 == 0: print generation
        # 第一代的不进入活性文件
        fitness = calculate_normalized_fitness(population, age, score_list)
        mating_pool = make_mating_pool(population, fitness)
        try:
            score_list, population, active,active_score = reproduce(mating_pool, population_size, mutation_rate, mutate2, age)
        except Exception as e:
            continue
        # 每一步都写
        all_pop = pd.concat([pd.DataFrame(active), pd.DataFrame(active_score)], axis=1)
        all_pop.to_csv('./output/R-chemist_jnk_gsk_all_pop/' + str(i) + '_' + str(age) + '_every_all.csv', index=False,
                       header=False, mode='a')



        # every_active = pd.DataFrame(active)
        # every_active.to_csv('./output/30_age_50_population_4C_bad_mute_no_canonical.smi', index=False, header=False, mode='a')

        all_active_list.extend(active)
        # print(all_active_list)

        age += 1
        print('len of set: ', len(set(all_active_list)))

        #
        # print('len of list: ', len(all_active_list))
        # print('len of set: ', len(set(all_active_list)))

    # print(i, max_score[0], max_score[1], Chem.MolFromSmiles(max_score[1]).GetNumAtoms())
    # # print(max_score[1])
    #
    # results.append(max_score[0])
    # size.append(Chem.MolFromSmiles(max_score[1]).GetNumAtoms())
    #
    # print('len of list: ', len(all_active_list))
    # print('len of set: ', len(set(all_active_list)))
    # if len(set(all_active_list)) > 1000000:
    #     all_active_list = pd.DataFrame(set(all_active_list))
    #     all_active_list.to_csv('./output/30_age_50_population_4C_bad_mute_no_canonical.smi', index=False, header=False)
    #     break

# t1 = time.time()
# print('')
# print('time ', t1 - t0)
# print(max(results), np.array(results).mean(), np.array(results).std())
# print(max(size), np.array(size).mean(), np.array(size).std())

# torch.cuda.empty_cache()

