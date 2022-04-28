import awkward
import os.path as osp
import os
import glob
import torch
import awkward as ak
import time
import uproot
import uproot3
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataListLoader, DataLoader
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv, global_mean_pool, DataParallel, EdgeConv
from torch_geometric.data import Data
from torchsummary import summary
from sklearn.neighbors import kneighbors_graph
import scipy.sparse as ss
from datetime import datetime, timedelta
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import math
from torch.autograd import Function
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def create_graph_train(z, k, d, p1, p2, label):
    vec = []
    vec.append(np.array([d, z, k]).T)
    vec = np.array(vec)
    vec = np.squeeze(vec)

    v1 = [[ind, x] for ind, x in enumerate(p1) if x > -1]
    v2 = [[ind, x] for ind, x in enumerate(p2) if x > -1]

    a1 = np.reshape(v1,(len(v1),2)).T
    a2 = np.reshape(v2,(len(v2),2)).T
    edge1 = np.concatenate((a1[0], a2[0], a1[1], a2[1]),axis = 0)
    edge2 = np.concatenate((a1[1], a2[1], a1[0], a2[0]),axis = 0)
    edge = torch.tensor(np.array([edge1, edge2]), dtype=torch.long)
    return Data(x=torch.tensor(vec, dtype=torch.float), edge_index=edge, y=torch.tensor(label, dtype=torch.float))

def create_train_dataset_fulld(z, k, d, p1, p2, label):
    graphs = [create_graph_train(a, b, c, d, e, f) for a, b, c, d, e, f in zip(z, k, d, p1, p2, label)]
    #graphs.append(Data(x=torch.tensor(vec, dtype=torch.float), edge_index=edge, y=torch.tensor(label[i], dtype=torch.float)))
    return graphs

def create_train_dataset_fulld(z, k, d, p1, p2, label):
    graphs = []
    for i in range(len(z)):
#        if i%1000 == 0:
#            print("Processing event {}/{}".format(i, len(z)), end="\r")
        vec = []
        vec.append(np.array([d[i], z[i], k[i]]).T)
        vec = np.array(vec)
        vec = np.squeeze(vec)

        v1 = [[ind, x] for ind, x in enumerate(p1[i]) if x > -1]
        v2 = [[ind, x] for ind, x in enumerate(p2[i]) if x > -1]

        a1 = np.reshape(v1,(len(v1),2)).T
        a2 = np.reshape(v2,(len(v2),2)).T
        edge1 = np.concatenate((a1[0], a2[0], a1[1], a2[1]),axis = 0)
        edge2 = np.concatenate((a1[1], a2[1], a1[0], a2[0]),axis = 0)
        edge = torch.tensor(np.array([edge1, edge2]), dtype=torch.long)
        graphs.append(Data(x=torch.tensor(vec, dtype=torch.float), edge_index=edge, y=torch.tensor(label[i], dtype=torch.float)))
    return graphs

def create_test_dataset_fulld(z, k, d, p1, p2):
    graphs = []
    for i in range(len(z)):
        vec = []
        vec.append(np.array([d[i], z[i], k[i]]).T)
        vec = np.array(vec)
        vec = np.squeeze(vec)
        v1 = [[ind, x] for ind, x in enumerate(p1[i]) if x > -1]
        v2 = [[ind, x] for ind, x in enumerate(p2[i]) if x > -1]

        a1 = np.reshape(v1,(len(v1),2)).T
        a2 = np.reshape(v2,(len(v2),2)).T
        edge1 = np.concatenate((a1[0], a2[0], a1[1], a2[1]),axis = 0)
        edge2 = np.concatenate((a1[1], a2[1], a1[0], a2[0]),axis = 0)
        edge = torch.tensor(np.array([edge1, edge2]), dtype=torch.long)
        graphs.append(Data(x=torch.tensor(vec, dtype=torch.float), edge_index=edge))
    return graphs

def create_train_dataset_prmy(z, k, d, label):
    graphs = []
    for i in range(len(z)):
        vec = []
        vec.append(np.array([d[i], z[i], k[i]]).T)
        vec = np.array(vec)
        vec = np.squeeze(vec)
        ya = kneighbors_graph(vec, n_neighbors=int(np.floor(vec.shape[0]/2)))
        edges = np.array([ya.nonzero()[0], ya.nonzero()[1]])
        edge = torch.tensor(edges, dtype=torch.long)
        graphs.append(Data(x=torch.tensor(vec, dtype=torch.float), edge_index=edge, y=torch.tensor(label[i], dtype=torch.float)))
    return graphs

def create_test_dataset_prmy(z, k, d):
    graphs = []
    for i in range(len(z)):
        vec = []
        vec.append(np.array([d[i], z[i], k[i]]).T)
        vec = np.array(vec)
        vec = np.squeeze(vec)
        ya = kneighbors_graph(vec, n_neighbors=int(np.floor(vec.shape[0]/2)))
        edges = np.array([ya.nonzero()[0], ya.nonzero()[1]])
        edge = torch.tensor(edges, dtype=torch.long)
        graphs.append(Data(x=torch.tensor(vec, dtype=torch.float), edge_index=edge))
    return graphs

#Configuration from bash script
if "INFILE" in os.environ:
    infile_path = os.environ["INFILE"]
    model_name = os.environ["MODEL"]
    epochs = int(os.environ["EPOCHS"])

#Configuration in notebook
else:
    files = glob.glob("../ForJad/*train*.root")
    #files = glob.glob("/eos/user/r/riordan/data/wprime/*_train.root") + glob.glob("/eos/user/r/riordan/data/j3to9/*_train.root")
#    files = glob.glob("/sps/atlas/j/jzahredd/LJPTagger/GNN/data/wprime/*._000001.ANALYSIS.root_train.root") + glob.glob("/sps/atlas/j/jzahredd/LJPTagger/GNN/data/j3to9/*24603626*._000004.ANALYSIS.root_test.root")
    model_name = "GNN_full"
#    model_name = "LSTM"
#    model_name = "1DCNN"
#    model_name = "2DCNN"
#    model_name = "ImgCNN"
#    model_name = "GNN"
#    model_name = "Transformer"
#    nb_epochs = 20

#Load tf keras model
# jet_type = "Akt10RecoChargedJet" #track jets
jet_type = "Akt10UFOJet" #UFO jets

save_trained_model = True
intreename = "lundjets_InDetTrackParticles"

model_filename = "save/models/"+model_name+".hdf5"

def pad_ak(arr, l):
    arr = ak.pad_none(arr, l)
    arr = ak.fill_none(arr, 1)
    arr = ak.to_numpy(arr)
    return arr

def pad_ak3(arr, l):
    arr = ak.pad_none(arr, 1)
    arr = ak.fill_none(arr, [0])
    arr = arr[:,0]
    arr = ak.pad_none(arr, l)
    arr = ak.fill_none(arr, 0)
    #arr = ak.to_numpy(arr)
    return arr

print("Training tagger on files", len(files))
t_start = time.time()

dsids = np.array([])
NBHadrons = np.array([])
trial = np.ones((1,30))
all_lund_zs = ak.from_numpy(trial)
all_lund_kts = ak.from_numpy(trial)
all_lund_drs = ak.from_numpy(trial)
parent1 = ak.from_numpy(trial)
parent2 = ak.from_numpy(trial)
jet_pts = trial
jet_ms = trial
eta = np.array([])
vector = []
for file in files[0:1]:

    print("Loading file",file)

    infile = uproot.open(file)
    tree = infile[intreename]
    dsids = np.append( dsids, np.array(tree["DSID"].array()) )
    #print(tree.keys())
    #eta = ak.concatenate(eta, pad_ak3(tree["Akt10TruthJet_jetEta"].array(), 30),axis=0)
    mcweights = tree["mcWeight"].array()
    NBHadrons = np.append( NBHadrons, pad_ak(tree["Akt10TruthJet_inputJetGABHadrons"].array(), 30)[:,0] )
    parent1 = ak.concatenate((parent1, pad_ak3(tree["Akt10UFOJet_jetLundIDParent1"].array(), 2)), axis = 0)
    parent2 = ak.concatenate((parent2, pad_ak3(tree["Akt10UFOJet_jetLundIDParent2"].array(), 2)), axis = 0)

    #Get jet kinematics
    jet_pts = np.append(jet_pts, pad_ak(tree["AntiKt10UFOCSSKSoftDropBeta100Zcut10JetsCalib_jetPt"].array(), 30)[:,0])
    #jet_etas = pad_ak(tree["AntiKt10UFOCSSKSoftDropBeta100Zcut10JetsCalib_jetEta"].array(), 30)[:,0]
    #jet_phis = pad_ak(tree["AntiKt10UFOCSSKSoftDropBeta100Zcut10JetsCalib_jetPhi"].array(), 30)[:,0]
    jet_ms = np.append(jet_ms, pad_ak(tree["AntiKt10UFOCSSKSoftDropBeta100Zcut10JetsCalib_jetM"].array(), 30)[:,0])

    #Get Lund variables
    all_lund_zs = ak.concatenate((all_lund_zs, pad_ak3(tree["{}_jetLundZ".format(jet_type)].array(), 2)), axis=0)
    all_lund_kts = ak.concatenate((all_lund_kts, pad_ak3(tree["{}_jetLundKt".format(jet_type)].array(), 2)), axis=0)
    all_lund_drs = ak.concatenate((all_lund_drs, pad_ak3(tree["{}_jetLundDeltaR".format(jet_type)].array(), 2)), axis=0)
    #print(len(jet_pts), len(jet_ms))
all_lund_zs = all_lund_zs[1:]
all_lund_kts = all_lund_kts[1:]
all_lund_drs = all_lund_drs[1:]
parent1 = parent1[1:]
parent2 = parent2[1:]
jet_pts = jet_pts[1:]
jet_ms = jet_ms[1:]


delta_t_fileax = time.time() - t_start
print("Opened data in {:.4f} seconds.".format(delta_t_fileax))

#Get labels
#labels = ( dsids > 360000 ) & ( dsids < 370000 )

labels = ( dsids > 370000 ) & ( NBHadrons == 0 ) # do NBHadrons == 0 for W bosons, NBHadrons > 0 for Tops

#print(labels)
labels = to_categorical(labels, 2)
labels = np.reshape(labels[:,1], (len(all_lund_zs), 1))

t_start = time.time()
#W bosons
dataset = create_train_dataset_fulld(all_lund_zs, all_lund_kts, all_lund_drs, parent1, parent2, labels)
#train_loader = DataLoader(dataset, batch_size=1024, shuffle=True)
dataset= shuffle(dataset,random_state=0)
train_ds, validation_ds = train_test_split(dataset, test_size = 0.2, random_state = 144)
train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
val_loader = DataLoader(validation_ds, batch_size=1024, shuffle=False)
delta_t_fileax = time.time() - t_start
print("Splitted datasets in {:.4f} seconds.".format(delta_t_fileax))


print ("train dataset size:", len(train_ds))
print ("validation dataset size:", len(validation_ds))

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = EdgeConv(nn.Sequential(nn.Linear(6, 64),
                                  nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64)),aggr='add')
        self.conv2 = EdgeConv(nn.Sequential(nn.Linear(128, 128),
                                  nn.ReLU(), nn.Linear(128, 128),nn.ReLU(), nn.Linear(128, 128)),aggr='add')
        self.conv3 = EdgeConv(nn.Sequential(nn.Linear(256,256,),
                                  nn.ReLU(), nn.Linear(256, 256),nn.ReLU(), nn.Linear(256, 256)),aggr='add')
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(256, 64)
        self.lin3 = torch.nn.Linear(64, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.1)
        x = self.lin2(x)
        x = F.dropout(x, p=0.1)
        x = self.lin3(x)
        #print(x.shape)
        return F.sigmoid(x)

class LundNet(torch.nn.Module):
    def __init__(self):
        super(LundNet, self).__init__()
        self.conv1 = EdgeConv(nn.Sequential(nn.Linear(6, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU(),
                                            nn.Linear(32, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU()),aggr='mean')
        self.conv2 = EdgeConv(nn.Sequential(nn.Linear(64, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU(),
                                            nn.Linear(32, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU()),aggr='mean')
        self.conv3 = EdgeConv(nn.Sequential(nn.Linear(64,64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU(),
                                            nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU()),aggr='mean')
        self.conv4 = EdgeConv(nn.Sequential(nn.Linear(128, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU(),
                                            nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU()),aggr='mean')
        self.conv5 = EdgeConv(nn.Sequential(nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU()),aggr='mean')
        self.conv6 = EdgeConv(nn.Sequential(nn.Linear(256, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU()),aggr='mean')
        self.sc1 = nn.Sequential(nn.Linear(3, 32, bias=False), nn.BatchNorm1d(num_features=32))
        self.sc2 = nn.Sequential(nn.ReLU())
        self.sc3 = nn.Sequential(nn.Linear(32, 64, bias=False), nn.BatchNorm1d(num_features=64))
        self.sc5 = nn.Sequential(nn.Linear(64, 128, bias=False), nn.BatchNorm1d(num_features=128))
        self.seq1 = nn.Sequential(nn.Linear(448, 384),
                                nn.BatchNorm1d(num_features=384),
                                nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(384, 256),
                                  nn.ReLU())
        self.lin = nn.Linear(256, 2)


path = "Models/LundNet_NewData_e002_0.15900.pt"

clsf = LundNet()
clsf.load_state_dict(torch.load(path))

ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)

class MDN(nn.Module):
    """A mixture density network layer
    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.
    Arguments:
        in_features (int): the number of dimensions in the input

        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions
    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.
    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """

    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=1)
        )
        self.sigma = nn.Linear(in_features, out_features * num_gaussians)
        self.mu = nn.Linear(in_features, out_features * num_gaussians)

    def forward(self, minibatch):
        pi = self.pi(minibatch)
        sigma = torch.exp(self.sigma(minibatch))
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu


def gaussian_probability(sigma, mu, target):
    """Returns the probability of `target` given MoG parameters `sigma` and `mu`.
    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        target (BxI): A batch of target. B is the batch size and I is the number of
            input dimensions.
    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
    target = target.unsqueeze(1).expand_as(sigma)
    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma)**2) / sigma
    ret = torch.where(ret == 0, ret + 1E-20, ret)
    return torch.prod(ret, 2)


def mdn_loss(pi, sigma, mu, target):
    """Calculates the error, given the MoG parameters and the target
    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    prob = pi * gaussian_probability(sigma, mu, target)
    nll = -torch.log(torch.sum(prob, dim=1))
    return torch.mean(nll)


def sample(pi, sigma, mu):
    """Draw samples from a MoG.
    """
    # Choose which gaussian we'll sample from
    pis = Categorical(pi).sample().view(pi.size(0), 1, 1)
    # Choose a random sample, one randn for batch X output dims
    # Do a (output dims)X(batch size) tensor here, so the broadcast works in
    # the next step, but we have to transpose back.
    gaussian_noise = torch.randn(
        (sigma.size(2), sigma.size(0)), requires_grad=False)
    variance_samples = sigma.gather(1, pis).detach().squeeze()
    mean_samples = mu.detach().gather(1, pis).squeeze()
    return (gaussian_noise * variance_samples + mean_samples).transpose(0, 1)

num_gaussians = 20 # number of Gaussians at the end
# The architecture of the adversary could be changed
class Adversary(nn.Module):
    def __init__(self):
        super(Adversary, self).__init__()
        self.gauss = nn.Sequential(
    nn.Linear(2, 64),
    nn.ReLU(),
    MDN(64, 1, num_gaussians)
)
        self.revgrad = GradientReversal(10)

    def forward(self, x):
        x = self.revgrad(x) # important hyperparameter, the scale,
                                     # tells by how much the classifier is punished
        x = self.gauss(x)
        return x

ms = np.array(jet_ms).reshape(len(jet_ms), 1)
pts = np.array(np.log(jet_pts)).reshape(len(jet_pts), 1)

def create_adversary_trainset(pt, mass):
    graphs = [Data(x=torch.tensor([p], dtype=torch.float), y=torch.tensor([m], dtype=torch.float)) for p, m in zip(pt, mass)]
    return graphs

class ConcatDataset(Dataset):
    def __init__(self, datasetA, datasetB):
        self.datasetA = datasetA
        self.datasetB = datasetB

    def __getitem__(self, index):
        xA = self.datasetA[index]
        xB = self.datasetB[index]
        return xA, xB

    def __len__(self):
        return len(self.datasetA)

adv_dataset = create_adversary_trainset(pts, ms)
conc_dataset = ConcatDataset(dataset, adv_dataset)
adv_loader = DataLoader(conc_dataset, batch_size=1024, shuffle=True)

adv = Adversary()
for param in clsf.parameters():
    param.require_grads = False

device = torch.device('cuda') # Usually gpu 4 worked best, it had the most memory available
clsf.to(device)
adv.to(device)
optimizer = torch.optim.Adam(adv.parameters(), lr=0.0005)
#optimizer = torch.optim.Adam(list(clsf.parameters()) + list(adv.parameters()), lr=0.0005)

def train(epoch):
    clsf.eval()
    adv.train()
    loss_all = 0
    for data in adv_loader:
        cl_data = data[0].to(device)
        adv_data = data[1].to(device)
        new_y = torch.reshape(cl_data.y, (int(list(cl_data.y.shape)[0]),1))
        mask_bkg = new_y.lt(0.5)
        optimizer.zero_grad()
        cl_out = clsf(cl_data)
        #print(torch.reshape(cl_out, (len(cl_out), 1)), torch.reshape(cl_out, (len(cl_out), 1)).shape)
        #print(adv_data.x, adv_data.x.shape)
        adv_inp = torch.cat((torch.reshape(cl_out[mask_bkg], (len(cl_out[mask_bkg]), 1)), torch.reshape(adv_data.x[mask_bkg], (len(adv_data.x[mask_bkg]), 1))), 1)
        #print(adv_inp.shape)
        pi, sigma, mu = adv(adv_inp)
        #cl_out = clsf(cl_data)
        loss2 = mdn_loss(pi, sigma, mu, torch.reshape(adv_data.y[mask_bkg], (len(cl_out[mask_bkg]), 1)))
        loss2.backward()
        loss_all += cl_data.num_graphs * loss2.item()
        optimizer.step()
    return loss_all / len(dataset)

@torch.no_grad()
def get_accuracy(loader):
    #remember to change this when evaluating combined model
    clsf.eval()
    correct = 0
    for data in loader:
        cl_data = data[0].to(device)
        #adv_data = data[1].to(device)
        new_y = torch.reshape(cl_data.y, (int(list(cl_data.y.shape)[0]),1))
        pred = clsf(cl_data).max(dim=1)[1]
        correct += pred.eq(new_y[:,1]).sum().item()
    return correct / len(loader.dataset)

@torch.no_grad()
def my_test (loader):
    clsf.eval()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        output = clsf(data)
        new_y = torch.reshape(data.y, (int(list(data.y.shape)[0]),2))
        loss = F.binary_cross_entropy(output, new_y)
        loss_all += data.num_graphs * loss.item()
    return loss_all/len(loader.dataset)

@torch.no_grad()
def get_scores(loader):
    clsf.eval()
    total_output = np.array([[1,1]])
    for data in loader:
        cl_data = data[0].to(device)
        pred = clsf(cl_data)
        total_output = np.append(total_output, pred.cpu().detach().numpy(), axis=0)
    return total_output[1:]


print("Training adversary whilst keeping classifier the same.")

for epoch in range(1, 3): # this may need to be bigger
    loss = train(epoch)
   # train_acc = get_accuracy(adv_loader)
    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: - '.format(epoch, loss))

for param in clsf.parameters():
    param.require_grads = True

device = torch.device('cuda') # Usually gpu 4 worked best, it had the most memory available
clsf.to(device)
adv.to(device)
optimizer_cl = torch.optim.Adam(clsf.parameters(), lr=0.01)
optimizer_adv = torch.optim.Adam(adv.parameters(), lr=0.00001)
#optimizer = torch.optim.Adam(list(clsf.parameters()) + list(adv.parameters()), lr=0.0005)


def train(epoch):
    clsf.train()
    adv.train()
    loss_all = 0
    for data in adv_loader:
        cl_data = data[0].to(device)
        adv_data = data[1].to(device)
        new_y = torch.reshape(cl_data.y, (int(list(cl_data.y.shape)[0]),1))
        mask_bkg = new_y.lt(0.5)
        optimizer_cl.zero_grad()
        optimizer_adv.zero_grad()
        cl_out = clsf(cl_data)
        adv_inp = torch.cat((torch.reshape(cl_out[mask_bkg], (len(cl_out[mask_bkg]), 1)), torch.reshape(adv_data.x[mask_bkg], (len(adv_data.x[mask_bkg]), 1))), 1)
        pi, sigma, mu = adv(adv_inp)
        loss1 = F.binary_cross_entropy(cl_out, new_y)
        loss2 = mdn_loss(pi, sigma, mu, torch.reshape(adv_data.y[mask_bkg], (len(adv_data.y[mask_bkg]), 1)))
        loss = loss1 + loss2
        loss.backward()
        loss_all += cl_data.num_graphs * loss.item()
        optimizer_cl.step()
        optimizer_adv.step()
    return loss_all / len(dataset)

print("Started training together!")

train_loss = []
val_loss = []
train_acc = []
val_acc = []
n_epochs = 2
save_every_epoch = True

for epoch in range(1, 3): # this may need to be bigger
    loss = train(epoch)
    #train_acc = get_accuracy(adv_loader)
    print("Epoch:{}".format(epoch))
    train_loss.append(train(train_loader))
    val_loss.append(my_test(val_loader))
    train_acc.append(get_accuracy(train_loader))
    val_acc.append(get_accuracy(val_loader))
    print('Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f}, Train Acc: {:.5f}, Val Acc: {:.5f}'.format(epoch, train_loss[epoch], val_loss[epoch], train_acc[epoch], val_acc[epoch]))
    if (save_every_epoch):
        torch.save(model.state_dict(), path+model_name+"e{:03d}".format(epoch+1)+"_{:.5f}".format(val_loss[epoch])+".pt")
    elif epoch == n_epochs-1:
        torch.save(model.state_dict(), path+model_name+"e{:03d}".format(epoch+1)+"_{:.5f}".format(val_loss[epoch])+".pt")

    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc:'.format(epoch, loss))

path = "" ## path to store txt files
metrics = pd.DataFrame({"Train_Loss":train_loss,"Val_Loss":val_loss,"Train_Acc":train_acc,"Val_Acc":val_acc})
metrics.to_csv(path+"losses_"+model_name+datetime.now().strftime("%d%m-%H%M")+".txt", index = False)

path = "/sps/atlas/j/jzahredd/LJPTagger/GNN/models/lundnet_2opt_2_5.pt"
torch.save(clsf.state_dict(), path)
