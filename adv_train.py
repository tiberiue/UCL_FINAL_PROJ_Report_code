import awkward
import os.path as osp
import os
import glob
import torch
import awkward as ak
from torch.utils.data import Dataset
import time
import uproot
import uproot3
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
#from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataListLoader, DataLoader
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv, global_mean_pool, DataParallel, EdgeConv,PNAConv
from torch_geometric.data import Data
from torch_geometric.utils import degree
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

def create_train_dataset_fulld_new(z, k, d, edge1, edge2, label):
    graphs = []
    for i in range(len(z)):
        if (len(edge1[i])== 0) or (len(edge2[i])== 0):
            continue
        else:
            edge = torch.tensor(np.array([edge1[i], edge2[i]]) , dtype=torch.long)
        vec = []
        vec.append(np.array([d[i], z[i], k[i]]).T)
        vec = np.array(vec)
        vec = np.squeeze(vec)
        graphs.append(Data(x=torch.tensor(vec, dtype=torch.float), edge_index=edge, y=torch.tensor(label[i], dtype=torch.float)))
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

#Configuration from bash script

#files = glob.glob("../ForJad/*train*.root")
#files = glob.glob("../OldSamples/*train*.root")


files = glob.glob("/sps/atlas/k/khandoga/MySamples90/*_train.root")
#files = files[:2]
#files = glob.glob("/sps/atlas/k/khandoga/MySamplesS40/user.rvinasco.27045978._000004.tree.root_train.root")
#files = files[:8]
jet_type = "Akt10UFOJet" #UFO jets
save_trained_model = True
intreename = "FlatSubstructureJetTree"

print("Training tagger on files", len(files))
t_start = time.time()

dsids = np.array([])
NBHadrons = np.array([])

all_lund_zs = np.array([])
all_lund_kts = np.array([])
all_lund_drs = np.array([])

parent1 = np.array([])
parent2 = np.array([])

jet_pts = np.array([])
jet_ms = np.array([])
jet_etas = np.array([])
jet_phis = np.array([])


eta = np.array([])
jet_truth_pts = np.array([])
jet_truth_etas = np.array([])
jet_truth_dRmatched = np.array([])
jet_truth_split = np.array([])
jet_ungroomed_ms = np.array([])
jet_ungroomed_pts = np.array([])
vector = []
mcweights = np.array([])
intreename = "FlatSubstructureJetTree"
for file in files:

    print("Loading file",file)
    with uproot.open(file) as infile:

        tree = infile[intreename]
        dsids = np.append( dsids, np.array(tree["DSID"].array()) )
        #eta = ak.concatenate(eta, pad_ak3(tree["Akt10TruthJet_jetEta"].array(), 30),axis=0)
        mcweights = np.append( mcweights, np.array(tree["mcWeight"].array()) )  
        NBHadrons = np.append( NBHadrons, ak.to_numpy(tree["Akt10UFOJet_GhostBHadronsFinalCount"].array()))

        parent1 = np.append(parent1, tree["UFO_edge1"].array(library="np"),axis=0)
        parent2 = np.append(parent2, tree["UFO_edge2"].array(library="np"),axis=0)

        #Get jet kinematics
        jet_pts = np.append(jet_pts, tree["UFOSD_jetPt"].array(library="np"))
        jet_etas = np.append(jet_etas, tree["UFOSD_jetEta"].array(library="np"))
        jet_phis = np.append(jet_phis, tree["UFOSD_jetPhi"].array(library="np"))

#        jet_truth_pts = np.append(jet_truth_pts, tree["Truth_jetPt"].array(library="np"))
#        jet_truth_etas = np.append(jet_truth_etas, ak.to_numpy(tree["Truth_jetEta"].array(library="np")))

        jet_ms = np.append(jet_ms, ak.to_numpy(tree["UFOSD_jetM"].array()))

        #Get Lund variables
        all_lund_zs = np.append(all_lund_zs,tree["UFO_jetLundz"].array(library="np") ) 
        all_lund_kts = np.append(all_lund_kts, tree["UFO_jetLundKt"].array(library="np") ) 
        all_lund_drs = np.append(all_lund_drs, tree["UFO_jetLundDeltaR"].array(library="np") )

#Get labels

labels = ( dsids > 370000 ) & ( NBHadrons == 0 ) # do NBHadrons == 0 for W bosons, NBHadrons > 0 for Tops

#print(labels)
labels = to_categorical(labels, 2)
labels = np.reshape(labels[:,1], (len(all_lund_zs), 1))

#W bosons
dataset = create_train_dataset_fulld_new(all_lund_zs, all_lund_kts, all_lund_drs, parent1, parent2, labels)
#train_loader = DataLoader(dataset, batch_size=1024, shuffle=True)
delta_t_fileax = time.time() - t_start
print("Created dataset in {:.4f} seconds.".format(delta_t_fileax))


deg = torch.zeros(10, dtype=torch.long)
for data in dataset:
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    deg += torch.bincount(d, minlength=deg.numel())



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
                                            nn.ReLU()),aggr='add')
        self.conv2 = EdgeConv(nn.Sequential(nn.Linear(64, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU(),
                                            nn.Linear(32, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU()),aggr='add')
        self.conv3 = EdgeConv(nn.Sequential(nn.Linear(64,64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU(),
                                            nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU()),aggr='add')
        self.conv4 = EdgeConv(nn.Sequential(nn.Linear(128, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU(),
                                            nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU()),aggr='add')
        self.conv5 = EdgeConv(nn.Sequential(nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU()),aggr='add')
        self.conv6 = EdgeConv(nn.Sequential(nn.Linear(256, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU()),aggr='add')

        self.seq1 = nn.Sequential(nn.Linear(448, 384),
                                nn.BatchNorm1d(num_features=384),
                                nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(384, 256),
                                  nn.ReLU())
        #self.lin1 = torch.nn.Linear(128, 128)
        #self.lin2 = torch.nn.Linear(448, 64)
        self.lin = nn.Linear(256, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.conv4(x3, edge_index)
        x5 = self.conv5(x4, edge_index)
        x6 = self.conv6(x5, edge_index)
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        x = self.seq1(x)
        x = global_mean_pool(x, batch)
        x = self.seq2(x)
        x = F.dropout(x, p=0.1)
        x = self.lin(x)
        #print(x.shape)
        return F.sigmoid(x)



class PNASimpleNet(torch.nn.Module):
    def __init__(self,in_channels):
        aggregators = ['sum','mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation',"linear",'inverse_linear']

        super(PNASimpleNet, self).__init__()
        self.conv1 = PNAConv(in_channels, out_channels=32, deg=deg, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.conv2 = PNAConv(in_channels=32, out_channels=64, deg=deg, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.conv3 = PNAConv(in_channels=64, out_channels=128, deg=deg, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.conv4 = PNAConv(in_channels=128, out_channels=256, deg=deg, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.conv5 = PNAConv(in_channels=256, out_channels=256, deg=deg, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.conv6 = PNAConv(in_channels=256, out_channels=256, deg=deg, post_layers=1,aggregators=aggregators,
                                            scalers = scalers)
        self.seq1 = nn.Sequential(nn.Linear(992, 384),
                                nn.BatchNorm1d(num_features=384),
                                nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(384, 256),
                                  nn.ReLU())
        self.lin = nn.Linear(256, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.conv4(x3, edge_index)
        x5 = self.conv5(x4, edge_index)
        x6 = self.conv6(x5, edge_index)
        x = torch.cat((x1, x2,x3,x4,x5,x6), dim=1)
        x = self.seq1(x)
        x = global_mean_pool(x, batch)
        x = self.seq2(x)
        x = F.dropout(x, p=0.1)
        x = self.lin(x)
        return F.sigmoid(x)


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
        self.bn1 = nn.BatchNorm1d(num_features=in_features)
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=1)
        )
        self.sigma = nn.Sequential(
            nn.Linear(in_features, out_features * num_gaussians),
            nn.Softplus()
        )

        self.mu = nn.Sequential(
            nn.Linear(in_features, out_features * num_gaussians),
            nn.Sigmoid()
        )
        #self.sigma = nn.Linear(in_features, out_features * num_gaussians)
        #self.mu = nn.Linear(in_features, out_features * num_gaussians)

    def forward(self, minibatch):
        minibatch = self.bn1(minibatch)
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
lambda_parameter = 10

# The architecture of the adversary could be changed
class Adversary(nn.Module):
    def __init__(self):
        super(Adversary, self).__init__()
        self.gauss = nn.Sequential(
    nn.Linear(2, 64),
    nn.ReLU(),
    MDN(64, 1, num_gaussians)
)
        self.revgrad = GradientReversal(lambda_parameter)

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

batch_size = 2048


test_ds, test1_ds = train_test_split(dataset, test_size = 0.2, random_state = 144)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

adv_dataset = create_adversary_trainset(pts, ms)
conc_dataset = ConcatDataset(dataset, adv_dataset)


conc_dataset= shuffle(conc_dataset,random_state=0)
train_ds, validation_ds = train_test_split(conc_dataset, test_size = 0.2, random_state = 144)


adv_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validation_ds, batch_size=batch_size, shuffle=False)

print ("train dataset size:", len(train_ds))
print ("validation dataset size:", len(validation_ds))


print ("Loading classifier model.")


path = "Models/LundNet_ufob_jsd_e022_0.16801.pt"

clsf = LundNet()
clsf.load_state_dict(torch.load(path))

print ("Classifier model loaded, training adversary.")


adv = Adversary()

#adv_model_weights = "/sps/atlas/k/khandoga/TrainGNN/Models/adv_e003_5.40484.pt"
#adv.load_state_dict(torch.load(adv_model_weights))


for param in clsf.parameters():
    param.require_grads = False

device = torch.device('cuda') # Usually gpu 4 worked best, it had the most memory available
clsf.to(device)
adv.to(device)
optimizer = torch.optim.Adam(adv.parameters(), lr=0.0005)
#optimizer = torch.optim.Adam(list(clsf.parameters()) + list(adv.parameters()), lr=0.0005)

def train_adversary(loader):
    clsf.eval()
    adv.train()
    loss_adv = 0
    loss_clsf = 0
    loss_all = 0
    batch_counter = 0 
 #   print ("batches in the dataset:", len(loader))
 #   print (" dataset length:", len(loader.dataset))
    for data in loader:
        batch_counter+=1
  #      print ("processing batch number",batch_counter)
        cl_data = data[0].to(device)
        adv_data = data[1].to(device)
        new_y = torch.reshape(cl_data.y, (int(list(cl_data.y.shape)[0]),1))

        mask_bkg = new_y.lt(0.5)
        optimizer.zero_grad()
        cl_out = clsf(cl_data)
        loss1 = F.binary_cross_entropy(cl_out, new_y)
        #print(torch.reshape(cl_out, (len(cl_out), 1)), torch.reshape(cl_out, (len(cl_out), 1)).shape)
        #print(adv_data.x, adv_data.x.shape)
        adv_inp = torch.cat((torch.reshape(cl_out[mask_bkg], (len(cl_out[mask_bkg]), 1)), torch.reshape(adv_data.x[mask_bkg], (len(adv_data.x[mask_bkg]), 1))), 1)
        #print(adv_inp.shape)
        pi, sigma, mu = adv(adv_inp)
        #cl_out = clsf(cl_data)
        loss2 = mdn_loss(pi, sigma, mu, torch.reshape(adv_data.y[mask_bkg], (len(cl_out[mask_bkg]), 1)))
        loss2.backward()
        loss = loss1 - 0.1*loss2
  #      print ("loss1",loss1.item())
  #      print ("loss2",loss2.item())
  #      print ("loss",loss.item())
        loss_clsf += cl_data.num_graphs * loss1.item()
        loss_adv += cl_data.num_graphs * loss2.item()
        loss_all += cl_data.num_graphs * loss.item()
        optimizer.step()
    return loss_adv / len(loader.dataset), loss_clsf / len(loader.dataset), loss_all / len(loader.dataset)

def train_combined(loader):
    clsf.train()
    adv.train()
    loss_adv = 0
    loss_clsf = 0
    loss_all = 0
    batch_counter = 0 
 #   print ("batches in the dataset:", len(loader))
    for data in loader:
        batch_counter+=1
 #       print ("processing batch number",batch_counter)
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
        loss = loss1 - 10*loss2
        loss.backward()
        loss_clsf += cl_data.num_graphs * loss1.item()
        loss_adv += cl_data.num_graphs * loss2.item()
        loss_all += cl_data.num_graphs * loss.item()
        optimizer_cl.step()
        optimizer_adv.step()
    return loss_adv / len(loader.dataset), loss_clsf / len(loader.dataset), loss_all / len(loader.dataset)


def test_combined(loader):
    clsf.eval()
    adv.eval()
    loss_adv = 0
    loss_clsf = 0
    loss_all = 0
    for data in loader:
        cl_data = data[0].to(device)
        adv_data = data[1].to(device)
        new_y = torch.reshape(cl_data.y, (int(list(cl_data.y.shape)[0]),1))
        mask_bkg = new_y.lt(0.5)
        cl_out = clsf(cl_data)
        adv_inp = torch.cat((torch.reshape(cl_out[mask_bkg], (len(cl_out[mask_bkg]), 1)), torch.reshape(adv_data.x[mask_bkg], (len(adv_data.x[mask_bkg]), 1))), 1)
        pi, sigma, mu = adv(adv_inp)
        loss1 = F.binary_cross_entropy(cl_out, new_y)
        loss2 = mdn_loss(pi, sigma, mu, torch.reshape(adv_data.y[mask_bkg], (len(adv_data.y[mask_bkg]), 1)))
        loss = loss1 - 10*loss2
        loss_clsf += cl_data.num_graphs * loss1.item()
        loss_adv += cl_data.num_graphs * loss2.item()
        loss_all += cl_data.num_graphs * loss.item()
    return loss_adv / len(loader.dataset), loss_clsf / len(loader.dataset), loss_all / len(loader.dataset)


@torch.no_grad()
def get_accuracy(loader):
    clsf.eval()
    correct = 0
    for data in loader:
        cl_data = data[0].to(device)
        #adv_data = data[1].to(device)
        new_y = torch.reshape(cl_data.y, (int(list(cl_data.y.shape)[0]),1))
        pred = clsf(cl_data).max(dim=1)[1]
        correct += pred.eq(new_y[0,:]).sum().item()
    return correct / len(loader.dataset)


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

train_loss_clsf = []
train_loss_adv = []
train_loss_total = []

val_loss_clsf = []
val_loss_adv = []
val_loss_total = []

train_acc = []
val_acc = []
path = "/sps/atlas/k/khandoga/TrainGNN/" ## path to store models txt files
model_name = "lund_adv_25_negloss_lambda10__"
metrics_filename = path+"Models/"+"losses_"+model_name+datetime.now().strftime("%d%m-%H%M")+".txt"

save_adv_every_epoch = True
adv_model_name = "adv_"

n_epochs_adv = 20

for epoch in range(n_epochs_adv): # this may need to be bigger
 #   my_test(test_loader)

    ad_lt, clsf_lt, total_lt =  train_adversary(adv_loader)
    train_loss_adv.append(ad_lt)
    train_loss_clsf.append(clsf_lt)
    train_loss_total.append(total_lt)
    train_acc.append(get_accuracy(adv_loader))

    ad_lv, clsf_lv, total_lv =  test_combined(val_loader)
    val_loss_adv.append(ad_lv)
    val_loss_clsf.append(clsf_lv)
    val_loss_total.append(total_lv)
    val_acc.append(get_accuracy(val_loader))

    print('Epoch: {:03d}, Train Loss total: {:.5f}, Train Loss adv: {:.5f}, Train Loss clsf: {:.5f}, val_loss_adv: {:.5f}, val_loss_clsf: {:.5f}, val_loss_total: {:.5f},train_acc: {:.5f},val_acc: {:.5f}'.format(epoch, train_loss_total[epoch],train_loss_adv[epoch],train_loss_clsf[epoch], val_loss_adv[epoch], val_loss_clsf[epoch], val_loss_total[epoch],train_acc[epoch],val_acc[epoch]))
    metrics = pd.DataFrame({"Train_Loss_adv":train_loss_adv,"Train_Loss_clsf":train_loss_clsf,"Train_Loss_total":train_loss_total,"Val_Loss_Adv":val_loss_adv,"Val_loss_Class":val_loss_clsf,"val_loss_total":val_loss_total, "Train_Acc":train_acc,"Val_Acc":val_acc})
    metrics.to_csv(metrics_filename, index = False)
    if (save_adv_every_epoch):
        torch.save(adv.state_dict(), path+"Models/"+adv_model_name+"e{:03d}".format(epoch+1)+"_{:.5f}".format(val_loss_adv[epoch])+".pt")
    elif epoch == n_epochs-1:
        torch.save(adv.state_dict(), path+"Models/"+adv_model_name+"e{:03d}".format(epoch+1)+"_{:.5f}".format(val_loss_adv[epoch])+".pt")




