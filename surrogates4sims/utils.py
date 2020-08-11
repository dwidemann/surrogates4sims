#AUTOGENERATED! DO NOT EDIT! File to edit: dev/04_utils.ipynb (unless otherwise specified).

__all__ = ['silu', 'SiLU', 'create_opt', 'create_one_cycle', 'find_lr', 'printNumModelParams', 'calcAccuracy', 'rmse',
           'writeMessage', 'plotSample', 'plotSampleWpredictionByChannel', 'plotSampleWprediction', 'curl', 'jacobian',
           'stream2uv', 'show', 'convertSimToImage', 'create_movie', 'pkl_save', 'pkl_load']

#Cell
import torch
import torch.nn as nn
from torch_lr_finder import LRFinder
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from torchvision.utils import make_grid, save_image
import matplotlib.animation as manimati
from matplotlib import animation, rc
from IPython.display import HTML
import pickle

def silu(input):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:

        SiLU(x) = x * sigmoid(x)
    '''
    return input * torch.sigmoid(input) # use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions

class SiLU(nn.Module):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:

        SiLU(x) = x * sigmoid(x)

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    References:
        -  Related paper:
        https://arxiv.org/pdf/1606.08415.pdf

    Examples:
        >>> m = silu()
        >>> input = torch.randn(2)
        >>> output = m(input)

    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return silu(x)

def create_opt(lr,model):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    return opt

def create_one_cycle(opt,max_lr,epochs,dataLoader):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer=opt,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=len(dataLoader))

def find_lr(model,opt,loss_func,device,dataLoader):
    lr_finder = LRFinder(model=model, optimizer=opt, criterion=loss_func, device=device)
    lr_finder.range_test(dataLoader, end_lr=100, num_iter=200)
    lr_finder.plot()
    # reset model & opt to their original weights
    lr_finder.reset()

def printNumModelParams(model):
    layers_req_grad = 0
    tot_layers = 0

    params_req_grad = 0
    tot_params = 0

    for param in model.named_parameters():
        #print(param[0])
        if (param[1].requires_grad):
            layers_req_grad += 1
            params_req_grad += param[1].nelement()
        tot_layers += 1
        tot_params += param[1].nelement()
    print("{0:,} layers require gradients (unfrozen) out of {1:,} layers".format(layers_req_grad, tot_layers))
    print("{0:,} parameters require gradients (unfrozen) out of {1:,} parameters".format(params_req_grad, tot_params))

def calcAccuracy(preds, labels):
    softedPreds = torch.softmax(preds,dim=1)
    classPreds = softedPreds.argmax(dim=1)
    totCorrect = (classPreds == labels).sum().item()
    totNum = labels.nelement()
    return totCorrect/totNum

def rmse(preds, labels):
    d = (preds - labels)**2
    d = d.mean()
    r = d.sqrt()
    return r

def writeMessage(msg, versionName):
    # Write to file.
    print(msg)
    myFile = open(versionName+".txt", "a")
    myFile.write(msg)
    myFile.write("\n")
    myFile.close()

def plotSample(X):
        plt.figure(figsize=(20,20))

        plt.subplot(211)
        title = 'Channel 0'
        plt.title(title)
        plt.imshow(X[0])
        plt.colorbar()

        plt.subplot(212)
        title = 'Channel 1'
        plt.title(title)
        plt.imshow(X[1])
        plt.colorbar()

def plotSampleWpredictionByChannel(sample, prediction):
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(20,20, forward=True)

    axs[0, 0].imshow(sample[0])
    axs[0, 0].set_title('Simulated Channel 0')
    axs[0, 1].imshow(prediction[0])
    axs[0, 1].set_title('Predicted Channel 0]')
    axs[1, 0].imshow(sample[1])
    axs[1, 0].set_title('Simulated Channel 1')
    axs[1, 1].imshow(prediction[1])
    axs[1, 1].set_title('Predicted Channel 1')
    #plt.subplots_adjust(wspace=0, hspace=0)
    # for ax in axs.flat:
    #     ax.set(xlabel='x-label', ylabel='y-label')

    # # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()

def plotSampleWprediction(sample,prediction):
    plt.figure(figsize=(20,20))
    A = np.vstack([sample[0], sample[1]])
    B = np.vstack([prediction[0], prediction[1]])
    C = np.hstack([A,B])
    plt.axis('off')
    plt.imshow(C)
    plt.colorbar()


def curl(X,device='cpu'):
    f1 = X[:,0,:,:]
    f2 = X[:,1,:,:]
    df1_dy = f1[:,1:,:] - f1[:,:-1,:]
    df1_dy = torch.cat([df1_dy,torch.zeros((df1_dy.shape[0],1,f1.shape[2])).to(device)], axis=1)
    df2_dx = f2[:,:,1:] - f2[:,:,:-1]
    df2_dx = torch.cat([df2_dx,torch.zeros((f2.shape[0],f2.shape[1],1)).to(device)], axis=2)
    c = df1_dy - df2_dx
    c = c[:,None,:,:]
    return c

def jacobian(X,device='cpu'):
    f1 = X[:,0,:,:]
    f2 = X[:,1,:,:]

    df1_dx = f1[:,:,1:] - f1[:,:,:-1]
    df1_dx = torch.cat([df1_dx,torch.zeros((f2.shape[0],f2.shape[1],1)).to(device)], axis=2)

    df1_dy = f1[:,1:,:] - f1[:,:-1,:]
    df1_dy = torch.cat([df1_dy,torch.zeros((df1_dy.shape[0],1,f1.shape[2])).to(device)], axis=1)

    df2_dx = f2[:,:,1:] - f2[:,:,:-1]
    df2_dx = torch.cat([df2_dx,torch.zeros((f2.shape[0],f2.shape[1],1)).to(device)], axis=2)

    df2_dy = f2[:,1:,:] - f2[:,:-1,:]
    df2_dy = torch.cat([df2_dy,torch.zeros((df1_dy.shape[0],1,f1.shape[2])).to(device)], axis=1)

    return torch.stack([df1_dx, df1_dy, df2_dx, df2_dy], axis=1)

# http://farside.ph.utexas.edu/teaching/336L/Fluidhtml/node69.html
# When creating the stream function, the second channel of X is not going to be used.
# It's there so we don't have to change the AE model code.
def stream2uv(X,device='cpu'):
    u = X[:,0,1:,:] - X[:,0,:-1,:]
    w = torch.unsqueeze(u[:,-1,:],axis=1)
    u = torch.cat([u,w],axis=1)
    v = X[:,0,:,1:] - X[:,0,:,:-1]
    w = torch.unsqueeze(u[:,:,-1],axis=2)
    v = torch.cat([v,w],axis=2)
    return torch.stack([u,v], axis=1)


def show(img,flip=False):
    npimg = img.numpy()
    if flip:
        npimg = np.flip(npimg)
    plt.figure(figsize=(40,20))
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

def convertSimToImage(X):
    # X = [frames,channels,h,w]
    mid = 128
    M = 255
    mx = X.max()
    mn = X.min()
    X = (X - mn)/(mx - mn)

    #C = np.uint8(M*B)
    C = (M*X).type(torch.uint8)

    if C.shape[1] == 2:
        out_shape = C.shape
        Xrgb = torch.zeros((out_shape[0],3,out_shape[2],out_shape[3])).type(torch.uint8)
        filler = mid*torch.ones(C.shape[2:]).type(torch.uint8)
        filler = filler.unsqueeze(axis=0)
        for idx, frame in enumerate(C):
            #Xrgb[idx] = torch.cat([frame[0].unsqueeze(axis=0),filler,frame[1].unsqueeze(axis=0)],axis=0)
            Xrgb[idx] = torch.cat([frame,filler],axis=0)
            #Xrgb[idx] = torch.cat([filler,frame],axis=0)
    else:
        Xrgb = C
    return Xrgb


def create_movie(Xrgb,outfile='sim.mp4'):
    ti = 0
    title = 'sim'
    u_mx = 255 #np.max(np.abs(Xrgb))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(title)
    cmap = plt.cm.ocean
    img = ax.imshow(np.transpose(Xrgb[0], (1,2,0)), cmap=cmap, vmin=0, vmax=u_mx)
    #plt.show()

    # initialization function: plot the background of each frame
    def init():
        img = ax.imshow(np.transpose(np.flip(Xrgb[0]), (1,2,0)), cmap=cmap, vmin=0, vmax=u_mx)
        return (fig,)

    # animation function. This is called sequentially
    def animate(i):
        img = ax.imshow(np.transpose(np.flip(Xrgb[i]), (1,2,0)), cmap=cmap, vmin=0, vmax=u_mx)
        return (fig,)


    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(Xrgb), interval=20, blit=True)
    anim.save(outfile, fps=30, extra_args=['-vcodec', 'libx264'])


def pkl_save(D,fn):
    with open(fn,'wb') as fid:
        pickle.dump(D,fid)

def pkl_load(fn):
    with open(fn,'rb') as fid:
        D = pickle.load(fid)
        return D
