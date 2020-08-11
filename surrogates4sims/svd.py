#AUTOGENERATED! DO NOT EDIT! File to edit: dev/05_experiment_SVD.ipynb (unless otherwise specified).

__all__ = ['MantaFlowSVDDataset']

#Cell
# --- Must haves ---
import os, sys
sys.path.append('..')
import torch
from torch.utils.data import Dataset, DataLoader
from .datasets import MantaFlowDataset, getSingleSim, createMantaFlowTrainTest
import numpy as np
import matplotlib.pyplot as plt
from .utils import rmse, create_movie, convertSimToImage
from tqdm import tqdm
import pickle

#Cell
class MantaFlowSVDDataset(Dataset):
    def __init__(self,
                 simFrames,
                 svdFile = '/home/widemann1/surrogates4sims/svd_out.npz',
                 numToKeep=np.infty,numComponents=512,transform=None, preprocess=True):

        if '.pkl' in simFrames:
            with open(simFrames,'rb') as fid:
                self.data = pickle.load(fid)
            if numToKeep < len(self.data):
                self.data = self.data[:numToKeep]
        else:
            if type(simFrames) == list:
                self.simFrames = simFrames
            else:
                self.simFrames = glob(os.path.join(simFrames,'*.npz'))

            self.numToKeep = numToKeep
            self.transform = transform
            self.svdFile = svdFile
            self.numComponents = numComponents
            self.simLen = 200

            out = np.load(svdFile)
            self.vh = out['arr_2'][:numComponents]

            self.sims = []
            for i in range(len(self.simFrames)//self.simLen):
                sim = getSingleSim(i)
                self.sims.append(sim)

            tmp = []

            if numToKeep < len(self.sims):
                self.sims = self.sims[:numToKeep]

            for sim in tqdm(self.sims):
                coeff_data = []
                for f in sim:
                    X,y = self.loadfile(f)

                    if preprocess:
                        X,y = self.preprocessFcn(X,y)
                    X = X.reshape(-1,1)
                    coeffs = (self.vh@X).squeeze()
                    #print(coeffs.shape)
                    #print(y.shape)
                    coeff_data.append((coeffs,y))

                diff_data = []
                for idx,c in enumerate(coeff_data[:-1]):
                    #print(c)
                    X0,y0 = c
                    X1,y1 = coeff_data[idx+1]
                    X = np.concatenate([X0,y1-y0])
                    #print(X.shape)
                    diff_data.append((X,X1))
                tmp.append(diff_data)

            self.data = [i for sublist in tmp for i in sublist]

    def loadfile(self,fn):
        A = np.load(fn)
        X = A['x'].astype('float32')
        X = np.rollaxis(X,-1)
        y = A['y'].astype('float32')
        return X,y

    def preprocessFcn(self,X,y):
        x_range = 11.953
        X /= x_range
        y_range = [[0.2, 0.8], [0.04, 0.12], [0.0, 199.0]]
        for i, ri in enumerate(y_range):
            y[i] = (y[i]-ri[0]) / (ri[1]-ri[0]) * 2 - 1
        return X,y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X,y = self.data[idx]
        return X,y