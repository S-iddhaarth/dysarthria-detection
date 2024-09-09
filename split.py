from torch.utils.data import random_split
import data_loader
import os
from utils.utility import seed_everything
from tqdm import tqdm

seed_everything(100)

path = 'data\\dysarthria'
SPLIT = 0.2

data = data_loader.DataLoader({"train":path,"label_encoding":{
                "F_Con":0,
                "F_Dys":1,
                "M_Con":2,
                "M_Dys":3
            }},split_mode=True)

total = len(data)
val_length = int(len(data)*SPLIT)
train_length = total - val_length

train_val_split = random_split(data,[train_length,val_length])
train_path = os.path.join(path,"train")
valid_path = os.path.join(path,"valid")

os.makedirs(train_path,exist_ok=True)
os.makedirs(valid_path,exist_ok=True)

for i in tqdm(train_val_split[1],total=len(train_val_split[1]),desc="generaing validation split"):
    if not i['audio'] == None:
        old = i["path"]
        new = old.split('\\')
        new[1] = "valid"
        make_dir = '\\'.join(new[:-1])
        os.makedirs(make_dir,exist_ok=True)
        new = '\\'.join(new)
        os.rename(old,new)
        
for i in tqdm(train_val_split[0],total=len(train_val_split[0]),desc="generaing train split"):
    if not i['audio'] == None:
        old = i["path"]
        new = old.split('\\')
        new[1] = "train"
        make_dir = '\\'.join(new[:-1])
        os.makedirs(make_dir,exist_ok=True)
        new = '\\'.join(new)
        os.rename(old,new)

os.rmdir(path)