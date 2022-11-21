import yaml
from pytorch_lightning.utilities.cli import instantiate_class
import os
import pickle
import torch
config_file = 'misc/solver_config.yaml'

conf = yaml.safe_load(open(config_file))

conf['solver']['init_args']['env'] = instantiate_class((), conf['solver']['init_args']['env']) 
solver = instantiate_class((),conf['solver'])

for this_dataset_dir in conf['dataset_dirs']:
    this_datasets = os.listdir(this_dataset_dir)
    for this_dataset in this_datasets:
        this_data = pickle.load(open(os.path.join(this_dataset_dir, this_dataset),'rb'))
        A = torch.from_numpy(-this_data['constraints'][:,:-1])
        b = torch.from_numpy(-this_data['constraints'][:,-1])
        c_train = torch.from_numpy(this_data['train_costs'])
        c_test = torch.from_numpy(this_data['test_costs'])
        batch_size= c_test.shape[0]
        y_test = solver(A[None,...].expand(batch_size,-1,-1), b[None,...].expand(batch_size,-1),c_test[:batch_size])
        batch_size = c_train.shape[0]
        y_train = solver(A[None,...].expand(batch_size,-1,-1), b[None,...].expand(batch_size,-1),c_train[:batch_size])
        this_data['y_train'] = y_train
        this_data['y_test'] = y_test
        with open(os.path.join(this_dataset_dir,this_dataset),'wb') as ofh:
            pickle.dump(this_data,ofh)

