import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
import os.path
import random
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

def load_dataset_para(N=30000,NP=1800):

#	obstacles=np.zeros((N,2800),dtype=np.float32)
#	for i in tqdm(range(0,N)):
#		temp=np.fromfile('../dataset2/obs_cloud/obc'+str(i)+'.dat')
#		temp=temp.reshape(int(len(temp)/2),2)
#		obstacles[i]=temp.flatten()

        obstacles = Parallel(n_jobs=multiprocessing.cpu_count()-1)(delayed(load_one_file)(i) for i in tqdm(range(N)))	
        return 	np.array(obstacles)

def load_one_file(i):
    temp=np.fromfile('../dataset2/obs_cloud/obc'+str(i)+'.dat')
    temp=temp.reshape(int(len(temp)/2),2)
    return temp.flatten()


def load_dataset(N=30000,NP=1800):
        obstacles=np.zeros((N,2800),dtype=np.float32)
        for i in tqdm(range(0,N)):
                temp=np.fromfile('../dataset2/obs_cloud/obc'+str(i)+'.dat')
                temp=temp.reshape(int(len(temp)/2),2)
                obstacles[i]=temp.flatten()
        return obstacles
if __name__=='__main__':
    print('ori')
    a = load_dataset(N=100)
    print(a)
    print('para')
    b = load_dataset_para(N=100)
    print(b)
    print((a-b).sum())
