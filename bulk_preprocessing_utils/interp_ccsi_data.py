import os, sys
from glob import glob
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.interpolate import interp2d, griddata
from multiprocessing import Pool

def loadfile(fn, channel=1,Nx=128,Ny=128):
    D = pd.read_csv(fn)
    x = D['X (m)'].values.astype('float32')
    y = D['Y (m)'].values.astype('float32')
    X = []
    columns = D.columns
    z = D[columns[channel]].values.astype('float32')
    grid_x, grid_y, grid_z = interpData(x,y,z,
                                        Nx,Ny,
                                        delta_x=None,nextPow2=None,
                                        method='linear')
    return grid_z.astype('float32')

def interpData(x,y,z,Nx=None,Ny=None,delta_x=None,nextPow2=False,method='linear'):
    '''
    This function takes 3 lists of points (x,y,z) and maps them to a 
    rectangular grid. Either Nx or Ny must be set or delta_x must be set. 
    e.g. 
    
    x = y = z = np.random.rand(30)
    grid_x, grid_y, grid_z = interpData(x,y,z,Nx=128,Ny=128)
    
    or 
    
    grid_x, grid_y, grid_z = interpData(x,y,z,delta_x=1e-3,nextPow2=True)
    '''
    
    eps = 1e-4 # needed to make sure that the interpolation does not have nans. 
    def _NextPowerOfTwo(number):
        # Returns next power of two following 'number'
        return np.ceil(np.log2(number))
    
    if Nx == None and Ny == None:
        assert delta_x != None
        delta_y = delta_x
        Nx = int((x.max() - x.min())/delta_x)
        Ny = int((y.max() - y.min())/delta_y)

    if nextPow2:
        Nx = 2**_NextPowerOfTwo(Nx)
        Ny = 2**_NextPowerOfTwo(Ny)
        
    grid_x, grid_y = np.mgrid[x.min()+eps:x.max()-eps:Nx*1j,y.min()+eps:y.max()-eps:Ny*1j]
    grid_z = griddata(np.array([x,y]).T, z, (grid_x, grid_y), method=method)
    return grid_x, grid_y, grid_z


def getInt(f):
    return int(f.split('_')[-1].replace('.csv',''))


if __name__ == '__main__':

    dataDir = sys.argv[1]
    outdir = sys.argv[2]
    Nx = int(sys.argv[4])
    Ny = Nx
    channel = int(sys.argv[3])
    outdir = '{}/channel_{}/gridsize_{}'.format(outdir,channel,Nx)    
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    folders = os.listdir(dataDir)
    #folders = ['033','034']
    print(folders)
    
    def process_folder(fd):
        out = []
        fns = glob(os.path.join(dataDir,fd,'*.csv'))
        L = np.argsort(list(map(getInt,fns)))
        orderedFiles = [fns[i] for i in L]

        out = list(map(lambda x: loadfile(x,channel=channel,Nx=Nx,Ny=Ny), orderedFiles))
        out = np.array(out) 
        with open(os.path.join(outdir,fd + '.pkl'),'wb') as fid:
            pickle.dump(out,fid)
    
    def mp_handler():
        numThreads = 25
        pool_manager = Pool(numThreads)
        pool_manager.map(process_folder, folders)
        
    mp_handler()



        

