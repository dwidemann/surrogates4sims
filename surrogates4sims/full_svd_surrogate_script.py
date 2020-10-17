# AUTOGENERATED! DO NOT EDIT! File to edit: 09_manta_full_svd_surrogate.ipynb (unless otherwise specified).

__all__ = ['main', 'invPreprocess', 'test_model_rel_err', 'reconFrameOnly', 'predictLatentVectors',
           'get_lin_and_svd_rel_err', 'MLP', 'trainEpoch', 'validEpoch', 'build_latents', 'LatentVectors', 'arg_parser']

# Cell
# --- Must haves ---
import os, sys, argparse
sys.path.append('..')

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.cuda as cuda
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from .mantaflowDatasets import MantaFlowDataset, getSingleSim, createMantaFlowTrainTest

from .utils import create_opt, create_one_cycle, find_lr, printNumModelParams, \
                                    rmse, writeMessage, plotSampleWprediction, plotSampleWpredictionByChannel, \
                                    plotSample, curl, jacobian, stream2uv, create_movie, convertSimToImage, \
                                    pkl_save, pkl_load, reconFrame, rel_err

#from surrogates4sims.models import Generator, Encoder, AE_no_P, AE_xhat_z, AE_xhat_zV2

from .train import trainEpoch, validEpoch

from .svd import MantaFlowSVDDataset

import numpy as np
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle

# Cell
try: from nbdev.imports import IN_NOTEBOOK
except: IN_NOTEBOOK=False
print("Running in notebook" if IN_NOTEBOOK else "Not running in notebook")

# Cell

def main(args):
    global p # to do: make this just p_shape and use p.shape because other function seem to just use its shape
    global x_mx, x_mn
    global device
    global test_data, test_sims

    # set the GPU
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    if args.gpu_ids:
        gpu_str_list = ','.join([str(i) for i in args.gpu_ids])
        os.environ["CUDA_VISIBLE_DEVICES"]= gpu_str_list
        versionName_with_args = versionName + '_GPUs{}'.format(gpu_str_list.replace(',',''))
    else:
        versionName_with_args = versionName

    print('='*82 + '\n\nRunning LIN experiments with command line arguments!\n\n'
                             + '-'*82 + '\n'*2)
    versionName_with_args += '_w{}_latentDim{}_hd{}_bz{}_epochs{}'.format(args.window,
                                                                             args.numComponents,
                                                                             hd,
                                                                             bz,
                                                                             epochs)
    print(versionName_with_args)
    print('\n' + '='*82 + '\n')
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu_ids else 'cpu')
    print('Using device:', device)

    train_data, test_data = build_latents(svd_vec_file, args)

    # instead of training
    # returning these unprocessed datasets for exploration in the interactive notebook
    if IN_NOTEBOOK:
        return train_data, test_data

    # reduce the dimensions of z down to the numComponents
    for idx,d in enumerate(train_data):
        X = d[0][:,:args.numComponents]
        p = d[1]
        train_data[idx] = (X,p)

    for idx,d in enumerate(test_data):
        X = d[0][:,:args.numComponents]
        p = d[1]
        test_data[idx] = (X,p)

    # get max/min along each latent vec dimension
    D = []
    for d in train_data:
        D.append(np.hstack(d))
    D = np.vstack(D)
    x_mx = np.max(D,axis=0)
    x_mn = np.min(D,axis=0)

    # build dataset of latent vectors
    trainDataset = LatentVectors(train_data,mx=x_mx,mn=x_mn,doPreprocess=True,w=args.window,simLen=200)
    testDataset = LatentVectors(test_data,mx=x_mx,mn=x_mn,doPreprocess=True,w=args.window,simLen=200)

    # dataloaders
    trainDataLoader = DataLoader(dataset=trainDataset, batch_size=bz, shuffle=True, drop_last=True)
    testDataLoader = DataLoader(dataset=testDataset, batch_size=bz)

    # build model
    X,y = next(iter(trainDataLoader))
    model = MLP(X, hiddenLayerSizes=hiddenLayers, activation=activation)
    printNumModelParams(model)
    if len(args.gpu_ids) > 1:
        model = nn.DataParallel(model)

    # training loop
    L = nn.MSELoss()

    max_lr = .001
    opt = torch.optim.Adam(model.parameters(), lr=max_lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=patience)

    versionName_with_args += '_lr{}'.format(str(max_lr))

    try:
        os.mkdir(cps)
    except:
        print("checkpoints directory already exists :)")


    # create a summary writer.
    train_writer = SummaryWriter(os.path.join(tensorboard_direc, versionName_with_args,'train'))
    test_writer = SummaryWriter(os.path.join(tensorboard_direc, versionName_with_args,'valid'))
    tensorboard_recorder_step = 0
    total_steps = 0

    model = model.to(device)
    writeMessage('---------- Started Training ----------', versionName_with_args)
    bestLoss = np.infty


    # loop over the dataset multiple times
    for epoch in tqdm(range(1, epochs+1)):

        writeMessage("\n--- Epoch {0}/{1} ---".format(epoch, epochs), versionName_with_args)

        model.train()
        trainLoss, tensorboard_recorder_step, total_steps = trainEpoch(trainDataLoader,
                                                                       train_writer, model, opt, L,
                                                                       rmse, lr_scheduler,
                                                                       tensorboard_rate, device,
                                                                       tensorboard_recorder_step, total_steps)

        writeMessage("trainLoss: {:.4e}".format(trainLoss),versionName_with_args)
        writeMessage("LR: {:.4e}".format(opt.param_groups[0]['lr']),versionName_with_args)
    #     if trainLoss < bestLoss:
    #         bestLoss = trainLoss
    #         writeMessage("Better trainLoss: {:.4e}, Saving models...".format(bestLoss),versionName)
    #         torch.save(model.state_dict(), os.path.join(cps,versionName))

        model.eval()
        valLoss = validEpoch(testDataLoader, test_writer, model, L, rmse, device, tensorboard_recorder_step)
        writeMessage("valLoss: {:.4e}".format(valLoss),versionName_with_args)

        #checkpoint progress
        if valLoss < bestLoss:
            bestLoss = valLoss
            writeMessage("\nBetter valLoss: {:.4e}, Saving models...".format(bestLoss),versionName_with_args)
            torch.save(model.state_dict(), os.path.join(cps,versionName_with_args))

        lr_scheduler.step(trainLoss)
        #lr_scheduler.step(valLoss)

        if opt.param_groups[0]['lr'] < 5e-8:
            break
    writeMessage('---------- Finished Training ----------', versionName_with_args)

    # best val loss model
    model.load_state_dict(torch.load(os.path.join(cps,versionName_with_args)))
    model = model.to(device)
    model.eval()

    # relative errors of latent vectors
    Err = []
    for idx in range(len(test_data)):
        e = test_model_rel_err(model,idx,test_data,True)
        Err.append(e)
    plt.savefig(os.path.join(tensorboard_direc,versionName_with_args,'LIN_rel_errors.pdf'))
    print('\nMean latent vector relative error: {}'.format(np.mean(Err)))

    # load dataset of simulation frames
    trainData, testData = createMantaFlowTrainTest(dataDirec,simLen,testSplit,seed)
    testDataset = MantaFlowDataset(testData, reverseXY=reverseXY,numToKeep=numSamplesToKeep, AE=False)
    testDataLoader = DataLoader(dataset=testDataset, batch_size=len(testDataset))
    X_test,p_test = next(iter(testDataLoader))
    c,h,width = X_test.shape[1:]
    test_sims = []
    for i in range(len(test_data[:int(np.nan_to_num(numSamplesToKeep, posinf=1e10)/200)])):
        A = X_test[simLen*i:simLen*(i+1),:]
        test_sims.append(A)
    test_sims = torch.stack(test_sims)

    # load svd vectors
    svd_data = pkl_load(SVDFn)
    svd_vecs = svd_data['spatialVecs'][:,:args.numComponents]

    # relative errors of simulation frames
    svd_rel_err = []
    lin_rel_err = []
    for test_ind in range(len(test_data[:int(np.nan_to_num(numSamplesToKeep, posinf=1e10)/200)])):
        Xhat_LIN, Xhat, ground_truth = get_lin_and_svd_rel_err(test_ind, model, svd_vecs)
        rel_err_svd_only = rel_err(ground_truth,Xhat)
        rel_err_lin = rel_err(ground_truth,Xhat_LIN)
        print('\n\nRelative Errors with/without Reconstructed Simulation Frames\n\n')
        print('Test Ind {} SVD: {} LIN: {}'.format(test_ind,rel_err_svd_only, rel_err_lin))
        svd_rel_err.append(rel_err_svd_only)
        lin_rel_err.append(rel_err_lin)


    results = {"LIN_z_mean_rel_error": Err,
               "svd_rel_err": svd_rel_err,
               "lin_rel_err": rel_err_lin}
    pickle.dump(results, open(os.path.join(tensorboard_direc,versionName_with_args,"results.p"), "wb"))



def invPreprocess(xnew):
    x = ((xnew/2)+.5)*(x_mx-x_mn) + x_mn
    return x


def test_model_rel_err(model,test_ind,test_data,doPlot=False):
    # last model
    idx = test_ind # choose one of the test samples
    testDataset = LatentVectors(test_data[idx:idx+1],doPreprocess=True,w=simLen-1,mx=x_mx,mn=x_mn)
    testDataLoader = DataLoader(dataset=testDataset, batch_size=1)
    X,y = next(iter(testDataLoader))
    X.shape, y.shape

    xhat = X.to(device).clone()
    out = []
    for idx in range(y.shape[1]):
        xhat = model(xhat).clone()
        xhat[:,:,-p.shape[1]:] = y[:,idx:idx+1,-p.shape[1]:]
        out.append(xhat)
    out = torch.stack(out).squeeze()

    yy = y.squeeze().to(device)
    err = []
    for i in range(out.shape[0]):
        label = invPreprocess(yy[i].detach().cpu().numpy())
        e =  label - invPreprocess(out[i].detach().cpu().numpy())
        err.append(np.linalg.norm(e)/np.linalg.norm(label))

    if doPlot:
        plt.plot(err)
        plt.title('Test Samples Relative Errors'.format(test_ind))

    return err

def reconFrameOnly(svd_vecs,coeffs):
    R = np.zeros(svd_vecs.shape[0],)
    for idx, c in enumerate(coeffs):
        R += c*svd_vecs[:,idx]
    R = R.reshape(2,128,96)
    return R

def predictLatentVectors(model,test_ind):
    # last model
    idx = test_ind # choose one of the test samples
    testDataset = LatentVectors(test_data[idx:idx+1],doPreprocess=True,w=simLen-1,mx=x_mx,mn=x_mn)
    testDataLoader = DataLoader(dataset=testDataset, batch_size=1)
    X,y = next(iter(testDataLoader))

    xhat = X.to(device).clone()
    out = []
    for idx in range(y.shape[1]):
        xhat = model(xhat).clone()
        xhat[:,:,-p.shape[1]:] = y[:,idx:idx+1,-p.shape[1]:]
        out.append(xhat)
    out = torch.stack(out).squeeze()
    return out

## turn the above into a function:
def get_lin_and_svd_rel_err(test_ind, model, svd_vecs):
    z = predictLatentVectors(model,test_ind)
    invPreProcZ = []
    for zz in z:
        invPreProcZ.append(invPreprocess(zz.detach().cpu().numpy()))
    invPreProcZ = np.array(invPreProcZ)

    # reconstruct the sim from the LIN latent vectors
    Xhat_LIN = []
    for zz in invPreProcZ:
        coeffs = zz[:args.numComponents]
        R = reconFrameOnly(svd_vecs,coeffs)
        Xhat_LIN.append(R)
    Xhat_LIN = np.array(Xhat_LIN)

    # reconstruct the sim from the ground truth latent vectors
    Xhat = []
    for zz in test_data[test_ind][0][1:]:
        coeffs = zz[:args.numComponents]
        R = reconFrameOnly(svd_vecs,coeffs)
        Xhat.append(R)
    Xhat = np.array(Xhat)

    ground_truth = np.array(test_sims[test_ind,1:,:,:,:])

    return Xhat_LIN, Xhat, ground_truth



class MLP(nn.Module):
    def __init__(self, X, hiddenLayerSizes = [1024], activation=nn.ELU()):
        super(MLP,self).__init__()

        self.activation = activation
        self.inputSize = X.shape[1:]
        self.modules = []
        self.modules.append(nn.Linear(np.prod(self.inputSize),hiddenLayerSizes[0]))
        self.modules.append(self.activation)
        for idx,sz in enumerate(hiddenLayerSizes[:-1]):
            self.modules.append(nn.Linear(hiddenLayerSizes[idx],hiddenLayerSizes[idx+1]))
            self.modules.append(self.activation)

        self.modules.append(nn.Linear(hiddenLayerSizes[-1],np.prod(self.inputSize)))
        self.layers = nn.Sequential(*self.modules)


    def forward(self,x):
        x = self.layers(x)
        return x



def trainEpoch(myDataLoader, tensorboard_writer, model, opt, loss,
           metric, lr_scheduler, tensorboard_rate, device,
           tensorboard_recorder_step, total_steps):
    running_loss = 0.0
    running_rmse = 0.0
    total_loss = 0.0
    for i, sampleBatch in enumerate(myDataLoader, start=1):

        # --- Main Training ---
        combined_loss = 0.

        # gpu
        X,y = sampleBatch[0],sampleBatch[1]
        X = X.to(device)
        y = y.to(device)

        # zero the parameter gradients
        opt.zero_grad()

        y_hat = X.clone()
        predictions = []
        for w_idx in range(args.window):
            y_hat = model(y_hat).clone()
            y_hat[:,:,-p.shape[1]:] = y[:,w_idx:w_idx+1,-p.shape[1]:]
            predictions.append(y_hat)
            combined_loss += loss(y_hat,y[:,w_idx:w_idx+1,:])
        combined_loss.backward()
        opt.step()

        # loss
        batch_loss = combined_loss.item()
        running_loss += batch_loss
        total_loss += batch_loss

        # --- Metrics Recording ---

        # metrics
        predictions = torch.stack(predictions)
        r = metric(y_hat, y)
        running_rmse += r

        # record lr change
        total_steps += 1
        tensorboard_writer.add_scalar(tag="LR", scalar_value=opt.param_groups[0]['lr'], global_step=total_steps)

        # tensorboard writes
        if (i % tensorboard_rate == 0):
            tensorboard_recorder_step += 1
            avg_running_loss = running_loss/tensorboard_rate
            avg_running_rmse = running_rmse/tensorboard_rate
            tensorboard_writer.add_scalar(tag="Loss", scalar_value=avg_running_loss, global_step=tensorboard_recorder_step)
            tensorboard_writer.add_scalar(tag=metric.__name__, scalar_value=avg_running_rmse, global_step=tensorboard_recorder_step)
            # reset running_loss for the next set of batches. (tensorboard_rate number of batches)
            running_loss = 0.0
            running_rmse = 0.0

    return total_loss/len(myDataLoader), tensorboard_recorder_step, total_steps


def validEpoch(myDataLoader, tensorboard_writer, model, loss, metric,
               device, tensorboard_recorder_step):
    running_loss = 0.0
    running_rmse = 0.0
    for i, sampleBatch in enumerate(myDataLoader, start=1):

        combined_loss = 0.
        # --- Metrics Recording ---

        # gpu
        X,y = sampleBatch[0],sampleBatch[1]
        X = X.to(device)
        y = y.to(device)

        # forward, no gradient calculations
        with torch.no_grad():
            y_hat = X.clone()
            predictions = []
            for w_idx in range(args.window):
                y_hat = model(y_hat).clone()
                y_hat[:,:,-p.shape[1]:] = y[:,w_idx:w_idx+1,-p.shape[1]:]
                predictions.append(y_hat)
                combined_loss += loss(y_hat,y[:,w_idx:w_idx+1,:])

        running_loss += combined_loss.item()

        # metrics
        predictions = torch.stack(predictions)
        r = metric(y_hat, y)
        running_rmse += r

    avg_running_loss = running_loss/len(myDataLoader)
    avg_running_rmse = running_rmse/len(myDataLoader)
    tensorboard_writer.add_scalar(tag="Loss", scalar_value=avg_running_loss, global_step=tensorboard_recorder_step)
    tensorboard_writer.add_scalar(tag=metric.__name__, scalar_value=avg_running_rmse, global_step=tensorboard_recorder_step)

    return avg_running_loss


def build_latents(svd_vec_file, args):
    '''
    Build Latent Vectors (Warning... The computation of building the latent vectors takes a loooong time.)

    If svd_vec_file exists already, the latent vectors will get loaded and this will run quickly
    '''
    if os.path.exists(svd_vec_file):
        data = pkl_load(svd_vec_file)
        train_data = data['train_data']
        test_data = data['test_data']
    else:
        user_desire = input("File path doesn't exist, enter 'y' to build latents from scratch? (Takes ~1 hour.)")
        if user_desire != 'y':
            sys.exit('specify a loadable latent vec file'+
                     ' or type y to create a new one at your specified location')
        svd_data = pkl_load(SVDFn)
        print(svd_data.keys())

        svd_vecs = svd_data['spatialVecs'][:,:args.numComponents]
        print(svd_vecs.shape)

        trainData, testData = createMantaFlowTrainTest(dataDirec,simLen,testSplit,seed)
        print((len(trainData),len(testData)))

        def createSVDdataset(trainData):

            # datasets may be smaller because: numSamplesToKeep
            # Be careful the default is for the data to be preprocessed. Therefore, we have to invPrecprocess if
            # we are looking at relative errors.
            trainDataset = MantaFlowDataset(trainData, reverseXY=reverseXY,numToKeep=numSamplesToKeep, AE=False)
            trainDataLoader = DataLoader(dataset=trainDataset, batch_size=len(trainDataset))
            X_train,p_train = next(iter(trainDataLoader))
            print(X_train.shape, p_train.shape)
            z_train = list(map(lambda x: reconFrame(svd_vecs, x, args.numComponents),X_train.numpy()))

            train_recons, latent_vec_train = zip(*z_train)
            train_recons = np.array(train_recons)
            latent_vec_train = np.array(latent_vec_train)

            v = np.arange(0,len(latent_vec_train),simLen)

            sims = []
            for idx in v:
                sims.append((latent_vec_train[idx:idx+simLen],p_train[idx:idx+simLen]))
            sims = np.array(sims)
            print('num_sims {}'.format(len(sims)))
            return sims

        train_data = createSVDdataset(trainData)
        test_data = createSVDdataset(testData)
        D = {'train_data':train_data,'test_data':test_data}
        pkl_save(D,svd_vec_file)
    return train_data, test_data


class LatentVectors(Dataset):
    def __init__(self, data, mx, mn, doPreprocess=False, w=1, simLen=200):
        self.data = data
        self.doPreprocess = doPreprocess
        self.simLen = simLen
        self.w = w
        self.mx = mx
        self.mn = mn

    def __len__(self):
        return self.simLen*len(self.data)

    def preprocess(self,x):
        xnew = 2*((x-self.mn)/(self.mx-self.mn) - .5)
        return xnew

    def invPreprocess(self,xnew):
        x = ((xnew/2)+.5)*(self.mx-self.mn) + self.mn
        return x

    def __getitem__(self, idx):
        q,r = np.divmod(idx,self.simLen)
        X,p = self.data[q]
        r_idx = np.random.randint(0,self.simLen-self.w)
        x = np.hstack([X[r_idx:r_idx+1],p[r_idx:r_idx+1]])
        #print(x.shape)
        y = np.hstack([X[r_idx+1:r_idx+self.w+1],p[r_idx+1:r_idx+self.w+1]])
        #print(y.shape)
        if self.doPreprocess:
            x = self.preprocess(x)
            y = self.preprocess(y)
        return x, y


def arg_parser():

    desc = 'Train and save LIN model, compute relative error of reconstructions.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--window', required=True, type=int, help='window size for LIN training')
    parser.add_argument('--numComponents', required=True, type=int, help=('latent vector size'+
                    ', this does not include p. so the vectors will be of size numComponents + len(p)'))
    parser.add_argument('--gpu_ids', nargs='+', type=int, help='GPU IDs: e.g., --gpu_ids 0 1')

    # GPU Numbers to use. Comma seprate them for multi-GPUs.
    #gpu_ids = "0"#,1,2,3"
    #w = 30
    #numComponents = 16

    return parser

if __name__ == '__main__':

    if IN_NOTEBOOK:
        # can enter "'--gpu_ids','0','1'" for multiple GPUs (e.g.)
        args = arg_parser().parse_args(['--window', '30', '--numComponents', '16', '--gpu_ids', '0'])
    else:
        args = arg_parser().parse_args()

    # the following variables can be made into command line arguments as needed
    # right now, we're just making w and numComponents command line args

    DEBUG = False

    # model name, for tensorboard recording and checkpointing purposes.
    versionName = "full_svd_manta_MLP"

    # path to load model weights.
    pretrained_path = None

    # rate at which to record metrics. (number of batches to average over when recording metrics, e.g. "every 5 batches")
    tensorboard_rate = 5

    # number of epochs to train. This is defined here so we can use the OneCycle LR Scheduler.
    epochs = 1000

    # Data Directory
    dataDirec = '/data/mantaFlowSim/data/smoke_pos21_size5_f200/v'
    reverseXY = False
    SVDFn = '/data/mantaFlowSim/data/smoke_pos21_size5_f200/svd/svd.pkl'
    svd_vec_file = '/data/mantaFlowSim/data/smoke_pos21_size5_f200/svd/mantaSVDvecs.pkl'

    # checkpoint directory
    cps = 'cps'
    tensorboard_direc = "tb"

    findLRs = True  # to do: set up the LR to be automatically tuned by the CLI
    patience = 1

    # hyper-params
    seed = 1234
    np.random.seed(seed)
    testSplit = .1
    bz = 64
    numSamplesToKeep = np.infty #if not debugging
    simLen = 200

    hiddenLayers = [128,128]
    hd ='_'.join(map(str,hiddenLayers))
    activation = nn.Tanh()

    if DEBUG:
        epochs = 2
        numSamplesToKeep = 200
        createDebugData = True



    if IN_NOTEBOOK:
        train_data, test_data = main(args)
    else:
        main(args)