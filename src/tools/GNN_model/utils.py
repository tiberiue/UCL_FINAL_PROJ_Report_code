import awkward
import os.path as osp
import os
import glob
import torch
import awkward as ak
import time
import yaml
import uproot
import uproot3
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
#from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataListLoader, DataLoader
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv, global_mean_pool, DataParallel, EdgeConv, GATConv, GINConv, PNAConv
from torch_geometric.data import Data
#from torchsummary import summary
#from tensorflow.keras.utils import to_categorical, plot_model
#from sklearn.neighbors import NearestNeighbors
#from sklearn.neighbors import kneighbors_graph
import scipy.sparse as ss
from datetime import datetime, timedelta
from torch_geometric.utils import degree
from scipy.stats import entropy
import math

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
from ..GNN_model.models import mdn_loss

def load_yaml(file_name):
    assert(os.path.exists(file_name))
    with open(file_name) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


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

def create_adversary_trainset(pt, mass):
    graphs = [Data(x=torch.tensor([p], dtype=torch.float), y=torch.tensor([m], dtype=torch.float)) for p, m in zip(pt, mass)]
    return graphs


def train(loader, model, device, optimizer):
    print ("dataset size:",len(loader.dataset))
    model.train()
    loss_all = 0
    batch_counter = 0
    for data in loader:
        batch_counter+=1
#        print ("processing batch number",batch_counter)
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        new_y = torch.reshape(data.y, (int(list(data.y.shape)[0]),1))
        loss = F.binary_cross_entropy(output, new_y)
        loss.backward()
#        print ("data.num_graphs",data.num_graphs)
#        print ("loss.item()",loss.item())
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(loader.dataset)


def train_adversary(loader, clsf, adv, optimizer, device, loss_parameter):
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
        loss = loss1 - loss_parameter*loss2
  #      print ("loss1",loss1.item())
  #      print ("loss2",loss2.item())
  #      print ("loss",loss.item())
        loss_clsf += cl_data.num_graphs * loss1.item()
        loss_adv += cl_data.num_graphs * loss2.item()
        loss_all += cl_data.num_graphs * loss.item()
        optimizer.step()
    return loss_adv / len(loader.dataset), loss_clsf / len(loader.dataset), loss_all / len(loader.dataset)

def train_combined(loader, clsf, adv, optimizer_cl, optimizer_adv, device, loss_parameter):
    clsf.train()
    adv.train()
    loss_adv = 0
    loss_clsf = 0
    loss_all = 0
    batch_counter = 0
    jsd_total = 0
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

        cl_out = cl_out.clamp(0, 1)
        cl_out[cl_out!=cl_out] = 0

        adv_inp = torch.cat((torch.reshape(cl_out[mask_bkg], (len(cl_out[mask_bkg]), 1)), torch.reshape(adv_data.x[mask_bkg], (len(adv_data.x[mask_bkg]), 1))), 1)
        pi, sigma, mu = adv(adv_inp)

        loss1 = F.binary_cross_entropy(cl_out, new_y)
        loss2 = mdn_loss(pi, sigma, mu, torch.reshape(adv_data.y[mask_bkg], (len(adv_data.y[mask_bkg]), 1)))
        loss = loss1 - loss_parameter*loss2
#        loss = loss1
        loss.backward()
        loss_clsf += cl_data.num_graphs * loss1.item()
        loss_adv += cl_data.num_graphs * loss2.item()
        loss_all += cl_data.num_graphs * loss.item()
        optimizer_cl.step()
        optimizer_adv.step()
#        mask_tag = cl_out.lt(0.5)
#        mask_untag = cl_out.ge(0.5)
#        p, _ = np.histogram(np.array(adv_data.y[mask_bkg&mask_tag].cpu()), bins=MASSBINS, density=1.)
#        f, _ = np.histogram(np.array(adv_data.y[mask_bkg&mask_untag].cpu()), bins=MASSBINS, density=1.)
#        jsd = JSD(p,f)
#        jsd_total +=jsd
#        print ("jsd",jsd)

    return loss_adv / len(loader.dataset), loss_clsf / len(loader.dataset), loss_all / len(loader.dataset)


def test_combined(loader, clsf, adv, device, loss_parameter):
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
        loss = loss1 - loss_parameter*loss2
        loss_clsf += cl_data.num_graphs * loss1.item()
        loss_adv += cl_data.num_graphs * loss2.item()
        loss_all += cl_data.num_graphs * loss.item()
    return loss_adv / len(loader.dataset), loss_clsf / len(loader.dataset), loss_all / len(loader.dataset)


@torch.no_grad()
def get_accuracy(loader, model, device):
    #remember to change this when evaluating combined model
    model.eval()
    correct = 0
    for data in loader:
        cl_data = data[0].to(device)
        new_y = torch.reshape(cl_data.y, (int(list(cl_data.y.shape)[0]),1))
        pred = model(cl_data).max(dim=1)[1]
        correct += pred.eq(new_y[0,:]).sum().item()
    return correct / len(loader.dataset)

@torch.no_grad()
def my_test (loader, model, device):
    model.eval()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        new_y = torch.reshape(data.y, (int(list(data.y.shape)[0]),1))
        loss = F.binary_cross_entropy(output, new_y)
        loss_all += data.num_graphs * loss.item()
    return loss_all/len(loader.dataset)

@torch.no_grad()
def get_scores(loader, model, device):
    model.eval()
    total_output = np.array([[1]])
    batch_counter = 0
    for data in loader:
        batch_counter+=1
        print ("Processing batch", batch_counter, "of",len(loader))
        data = data.to(device)
        pred = model(data)
        total_output = np.append(total_output, pred.cpu().detach().numpy(), axis=0)

    return total_output[1:]


def aux_metrics(loader, clsf, adv, device, MASSBINS):
    clsf.eval()
    adv.eval()
    counter = 0
    bkg_tagged = 0
    bkg_total = 0
    jsd_total = 0
    nans = 0
    jsd_counter = 0
    mass_tagged = np.array([])
    mass_untagged = np.array([])
    for data in loader:
        cl_data = data[0].to(device)
        adv_data = data[1].to(device)
        new_y = torch.reshape(cl_data.y, (int(list(cl_data.y.shape)[0]),1))
    #    print ("true labels",new_y)
        mask_bkg = new_y.lt(0.5)
    #    print ("mask_bkg",mask_bkg)
        cl_out = clsf(cl_data)
   #     print ("preds",cl_out)
        mask_tag = cl_out.lt(0.5)
        mask_untag = cl_out.ge(0.5)
 #       print ("mask_tag",mask_tag)
 #       print ("mask_untag",mask_untag)
 #       print ("sum bg",torch.count_nonzero(mask_bkg))
 #       print ("sum tag bg",torch.count_nonzero(mask_tag))
        bkg_tagged+=torch.count_nonzero(mask_untag&mask_bkg)
        bkg_total+=torch.count_nonzero(mask_bkg)
#        print ("adv_data.x",adv_data.y)
#        print ("mass tag",adv_data.y[mask_bkg&mask_tag])
#        print ("mass untag",adv_data.y[mask_bkg&mask_untag])
        p, _ = np.histogram(np.array(adv_data.y[mask_bkg&mask_tag].cpu()), bins=MASSBINS, density=1.)
        f, _ = np.histogram(np.array(adv_data.y[mask_bkg&mask_untag].cpu()), bins=MASSBINS, density=1.)
        jsd = JSD(p,f)
        if math.isnan(jsd):
            nans+=1
        else:
            jsd_total +=jsd
            jsd_counter+=1
  #      print ("jsd",jsd)
    if bkg_tagged:
        eff = bkg_total/bkg_tagged
    else:
        eff = bkg_total*0

    if jsd_counter:
        jsd_total = jsd_total/jsd_counter
    else:
        jsd_total = 0

    return float(eff.cpu()), jsd_total


def JSD (P, Q, base=2):
    """Compute Jensen-Shannon divergence (JSD) of two distribtions.
    From: [https://stackoverflow.com/a/27432724]

    Arguments:
        P: First distribution of variable as a numpy array.
        Q: Second distribution of variable as a numpy array.
        base: Logarithmic base to use when computing KL-divergence.

    Returns:
        Jensen-Shannon divergence of `P` and `Q`.
    """
    p = P / np.sum(P)
    q = Q / np.sum(Q)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m, base=base) + entropy(q, m, base=base))
