import json
import torch
import lightning as L

def get_secret_keys(path):
    with open(path) as f:
        return json.load(f)
    
def set_seed(seed=42):
    '''Sets seed of the entire run so experiment results are the same every time we run for reproducibility.
    '''
    L.seed_everything(seed)
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True #to solve http requests error