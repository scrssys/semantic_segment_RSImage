import numpy as np
import pandas as pd
import shutil
import time
import os
from config import Config
def get_csv_folds(path, d):
    df = pd.read_csv(path, index_col=0)
    m = df.max()[0] + 1
    train = [[] for i in range(m)]
    test = [[] for i in range(m)]

    folds = {}
    for i in range(m):
        fold_ids = list(df[df['fold'].isin([i])].index)
        folds.update({i: [n for n, l in enumerate(d) if l in fold_ids]})

    for k, v in folds.items():
        for i in range(m):
            if i != k:
                train[i].extend(v)
        test[k] = v

    return list(zip(np.array(train), np.array(test)))
def update_config(config, **kwargs):
    d = config._asdict()
    d.update(**kwargs)
    print(d)
    return Config(**d)
def save(path,network,jsonPath=None):

    folder=time.strftime("%Y%m%d%H%M", time.localtime())
    new_path=os.path.join(path,"history",folder+"_"+network)
    try:
        os.makedirs(new_path)
    except:
        new_path=new_path+"_"+str(np.random.randint(0,100))
        os.makedirs ( new_path )
    shutil.copy(os.path.join(path,"train.py"),new_path)
    if jsonPath is None:
        shutil.copy ( os.path.join ( path , "config.json" ) , new_path )
    else:
        shutil.copy ( jsonPath , new_path )