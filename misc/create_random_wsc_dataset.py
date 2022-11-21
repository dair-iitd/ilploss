import numpy as np 
from constraint_handler.constraint_handler_utils import sample_constraints, sample_offset_constraints_numpy
from itertools import product 
import os
import pickle

MAX_TRAIN_CONSTRAINTS = 10000
TEST_NUM_CONSTRAINTS = 1000 
output_dir = '/home/cse/phd/csz178057/hpcscratch/comboptnet/synthetic_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


epsilon_constant = 1e-8
params_dict = {}
params_dict['num_variables'] = [16,32,64,128,256]
params_dict['num_constraints'] = [1,2,4,8,16,32,64]
num_seeds = 10
np.random.seed(42)
params_dict['seed'] = np.random.randint(10000,size = num_seeds) 
params_dict['variable_range'] =[{'lb': 0, 'ub': 1},{'lb': -5,'ub': 5}]
params_dict['offset_sample_point'] = ["random_unif"]
params_dict['feasible_point'] = ["origin"]

all_args = list(map(dict, product(*[[(k, v) for v in vv] for k, vv in params_dict.items()]))) 

 
constraint_type = 'random_const'

for var_type in ['binary','dense']:
    this_output_dir = os.path.join(output_dir,constraint_type,var_type)
    if not os.path.exists(this_output_dir):
        os.makedirs(this_output_dir)

for i,this_args in enumerate(all_args):
    if this_args['variable_range']['ub'] - this_args['variable_range']['lb'] == 1:
        var_type = 'binary'
    else:
        var_type = 'dense'
    #
    this_output_dir = os.path.join(output_dir,constraint_type,var_type)
    this_constraints = sample_constraints(constraint_type,**this_args)
    num_train = min(this_args['num_variables']*100,MAX_TRAIN_CONSTRAINTS)
    num_test = TEST_NUM_CONSTRAINTS 
    train_c = np.random.rand(num_train, this_args['num_variables']) - 0.5
    #train_c_norm = A / (np.linalg.norm(train_c, axis=1, keepdims=True) + epsilon_constant)
    test_c = np.random.rand(num_test, this_args['num_variables']) - 0.5
    #test_c_norm = A / (np.linalg.norm(test_c, axis=1, keepdims=True) + epsilon_constant)
    this_output_file = 'vars-{}_cons-{}_seed-{}.pkl'.format(this_args['num_variables'], this_args['num_constraints'], this_args['seed'])
    this_output_file = os.path.join(this_output_dir, this_output_file)
    this_output_dict = {'constraints': this_constraints, 'train_costs': train_c, 'test_costs': test_c}
    print("{} Dumping {} vars {} const {} train {} test {} seed in  {} ".format(
        var_type, this_args['num_variables'], this_args['num_constraints'], 
        num_train, num_test, 
        this_args['seed'],
        this_output_file))
    #
    with open(this_output_file,'wb') as ofh:
       pickle.dump(this_output_dict, ofh) 


constraint_type = 'set_covering'
#num_variables == size of the subset of powerset = 2*num_constraints
#num_constraints == #of elements: size of the universe 
#max_subset_size == fixed at 3 in the paper

del params_dict['offset_sample_point']
del params_dict['feasible_point']
del params_dict['num_variables']
params_dict['num_constraints'] = [8,9,10,16,20,25] #size of the universe. Cant increase it exponentially: the powerset is already exponential to num_constraints
params_dict['variable_range'] =[{'lb': 0, 'ub': 1}]
params_dict['max_subset_size'] = [3]
all_args = list(map(dict, product(*[[(k, v) for v in vv] for k, vv in params_dict.items()]))) 

this_output_dir = os.path.join(output_dir,constraint_type)
if not os.path.exists(this_output_dir):
    os.makedirs(this_output_dir)

for i,this_args in enumerate(all_args):
    this_args['num_variables'] = 2*this_args['num_constraints']
    this_constraints = sample_constraints(constraint_type,**this_args)
    num_train = min(this_args['num_variables']*100,MAX_TRAIN_CONSTRAINTS)
    num_test = TEST_NUM_CONSTRAINTS 
    train_c = np.random.rand(num_train, this_args['num_variables'])
    #train_c_norm = A / (np.linalg.norm(train_c, axis=1, keepdims=True) + epsilon_constant)
    test_c = np.random.rand(num_test, this_args['num_variables']) 
    #test_c_norm = A / (np.linalg.norm(test_c, axis=1, keepdims=True) + epsilon_constant)
    this_output_file = 'vars-{}_cons-{}_seed-{}.pkl'.format(this_args['num_variables'], this_args['num_constraints'], this_args['seed'])
    this_output_file = os.path.join(this_output_dir, this_output_file)
    this_output_dict = {'constraints': this_constraints, 'train_costs': train_c, 'test_costs': test_c}
    print("Dumping {} vars {} const {} train {} test {} seed in  {} ".format(
        this_args['num_variables'], this_args['num_constraints'], 
        num_train, num_test, 
        this_args['seed'],
        this_output_file))
    #
    with open(this_output_file,'wb') as ofh:
       pickle.dump(this_output_dict, ofh) 

