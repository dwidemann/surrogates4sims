# AUTOGENERATED! DO NOT EDIT! File to edit: 01_mantaflow_dataset.ipynb (unless otherwise specified).

__all__ = ['loadfile', 'MantaFlowDataset', 'getSingleSim', 'createMantaFlowTrainTest']

# Cell
#from torchvision import datasets, transforms
import os, sys
from glob import glob
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from random import shuffle
from tqdm import tqdm

# Cell
def loadfile(fn):
    A = np.load(fn)
    X = A['x'].astype('float32')
    X = np.rollaxis(X,-1)
    y = A['y'].astype('float32')
    return X,y

# Cell
class MantaFlowDataset(Dataset):
    def __init__(self,
                 dataDirec='/home/widemann1/carbon_capture/surrogate_nn_for_pde/deep-fluids/data/smoke_pos21_size5_f200/v',
                 numToKeep=np.infty,transform=None, reverseXY=False, preprocess=True, AE=False):
        if type(dataDirec) == list:
            self.files = dataDirec
        else:
            self.files = glob(os.path.join(dataDirec,'*.npz'))
        self.dataDirec = dataDirec
        self.numToKeep = numToKeep
        self.transform = transform
        self.reverseXY = reverseXY
        self.AE = AE
        self.data = []

        if numToKeep < len(self.files):
            self.files = self.files[:numToKeep]
        for f in tqdm(self.files):
            X,y = self.loadfile(f)

            if preprocess:
                X,y = self.preprocessFcn(X,y)

            if reverseXY:
                self.data.append((y,X))
            else:
                self.data.append((X,y))

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

    def plot(self,idx,savefig=False):
        X, label  = self.data[idx]
        if self.reverseXY:
            X = label

        plt.figure(figsize=(20,10))

        plt.subplot(211)
        fn = self.files[idx].replace('.npz','')
        title = '{} channel 0'.format(fn)
        plt.title(title)
        plt.imshow(X[0][::-1])
        plt.colorbar()

        plt.subplot(212)
        title = '{} channel 1'.format(fn)
        plt.title(title)
        plt.imshow(X[1][::-1])
        plt.colorbar()

        if savefig:
            title = title.replace(' ','_') + '.png'
            plt.savefig(title, dpi=300)
            plt.close()
        else:
            plt.show()

    def __getitem__(self, idx):
        X, y  = self.data[idx]
        if self.transform:
            X = self.transform(X)
        if self.AE:
            return X, X # looks crazy, but it's dirty fix to allow lr_finder to work
        return X, y

# Cell
import os
from glob import glob

def getSingleSim(sim,
                 dataDirec='/home/widemann1/carbon_capture/surrogate_nn_for_pde/deep-fluids/data/smoke_pos21_size5_f200/v',
                 simLen=200):
    if type(dataDirec) == str:
        data = glob(os.path.join(dataDirec,'*.npz'))
    else:
        data = dataDirec
    data = sorted(data)
    out = data[sim*simLen:(sim+1)*simLen]
    order = []
    for fn in out:
        a = os.path.basename(fn).replace('.npz','').split('_')
        order.append(int(a[2]))
    sorted_idx = np.argsort(order)
    out = [out[i] for i in sorted_idx]
    return out


def createMantaFlowTrainTest(dataDirec='/home/widemann1/carbon_capture/surrogate_nn_for_pde/deep-fluids/data/smoke_pos21_size5_f200/v',
                            simLen=200,
                            testSplit=.1,
                            seed=1234):
    data = glob(os.path.join(dataDirec,'*.npz'))
    data = sorted(data)
    numSims = len(data)//simLen
    numTestSamples = int(np.round(testSplit*numSims))
    np.random.seed(seed)
    perm = np.random.permutation(numSims)
    testSims = perm[:numTestSamples]
    trainSims = perm[numTestSamples:]

    def _getSimFrames(data, sim=0,simLen=200):
        out = data[sim*simLen:(sim+1)*simLen]
        order = []
        for fn in out:
            a = os.path.basename(fn).replace('.npz','').split('_')
            order.append(int(a[2]))
        sorted_idx = np.argsort(order)
        out = [out[i] for i in sorted_idx]
        return out

    def _buildDataset(simData, sims, simLen):
        data = []
        for i in sims:
            sim = _getSimFrames(simData, i, simLen)
            data.append(sim)
        data = [i for sublist in data for i in sublist]
        return data

    testData = _buildDataset(data, testSims, simLen)
    trainData = _buildDataset(data, trainSims, simLen)
    return trainData, testData