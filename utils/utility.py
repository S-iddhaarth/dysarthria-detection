import random
import numpy as np
import torch
import os 
import csv

def seed_everything(seed: int):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
def csv_to_tuples(file_path):
    with open(file_path, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        # Convert each row to a tuple and collect them in a list
        tuples_list = [tuple(row) for row in reader]
    return tuples_list[1:]