import argparse
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
import yaml
import scipy.sparse as ss
from tqdm.notebook import tqdm
from datetime import datetime, timedelta
from torch_geometric.utils import degree
from torch_geometric.data import DataListLoader, DataLoader

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd

from tools.GNN_model_weight.models import *
from tools.GNN_model_weight.utils  import *
import matplotlib.pyplot as plt



print("Libraries loaded!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train with configurations')
    add_arg = parser.add_argument
    add_arg('config', help="job configuration")
    args = parser.parse_args()
    config_file = args.config
    config = load_yaml(config_file)

    path_to_file = config['data']['path_to_files']
    files = glob.glob(path_to_file)
    #files = glob.glob("/sps/atlas/k/khandoga/MySamplesS40/user.rvinasco.27045978._000004.tree.root_train.root")
    #files = files[:1]
    jet_type = "Akt10UFOJet" #UFO jets
    
    intreename = "FlatSubstructureJetTree"

    print("Opening this many files:", len(files))
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
    eta = np.array([])
    jet_truth_pts = np.array([])
    jet_truth_etas = np.array([])
    jet_truth_dRmatched = np.array([])
    jet_truth_split = np.array([])
    jet_ungroomed_ms = np.array([])
    jet_ungroomed_pts = np.array([])
    vector = []
    for file in files:

        print("Loading file",file)

        with uproot.open(file) as infile:
            tree = infile[intreename]
            dsids = np.append( dsids, np.array(tree["DSID"].array()) )
            #eta = ak.concatenate(eta, pad_ak3(tree["Akt10TruthJet_jetEta"].array(), 30),axis=0)
            mcweights = tree["mcWeight"].array()
            NBHadrons = np.append( NBHadrons, ak.to_numpy(tree["Akt10UFOJet_GhostBHadronsFinalCount"].array()))

            parent1 = np.append(parent1, tree["UFO_edge1"].array(library="np"),axis=0)
            parent2 = np.append(parent2, tree["UFO_edge2"].array(library="np"),axis=0)
            jet_ms = np.append(jet_ms, ak.to_numpy(tree["UFOSD_jetM"].array()))
            jet_pts = np.append(jet_pts, ak.to_numpy(tree["UFOSD_jetPt"].array()))

            #Get jet kinematics

    #        jet_truth_split = np.append(jet_truth_split, tree["Akt10TruthJet_ungroomed_truthJet_Split12"].array(library="np"))

            #Get Lund variables .array(library="np")
            lundzs = tree["UFO_jetLundz"].array(library="np")
            lundkts = tree["UFO_jetLundKt"].array(library="np")
            lunddrs = tree["UFO_jetLundDeltaR"].array(library="np")
            all_lund_zs = np.concatenate(lundzs,axis = 0)
            print("z concatenated")
            all_lund_kts = np.concatenate(lundkts,axis = 0)
            print("kt concatenated")
            all_lund_drs = np.concatenate(lunddrs,axis = 0)
            print("dr concatenated")
            #for i in range(len()):
                
                #all_lund_zs = np.append(all_lund_zs,lundzs[i])
                #all_lund_kts = np.append(all_lund_kts,lundkts[i] )
                #all_lund_drs = np.append(all_lund_drs, lunddrs[i] )
                #print("loop",i,"out of",len(lundzs),"completed")
            print(all_lund_zs)
            print("z-shape",all_lund_zs.shape)
            print(all_lund_kts)
            print("kts-shape",all_lund_kts.shape)
            print(all_lund_drs)
            print("drs-shape",all_lund_drs.shape)
            
    ln1z = np.log(1/all_lund_zs)
    lnkt = np.log(all_lund_kts)
    #utilize this one for large jets    
    lnRDR = np.log(1/all_lund_drs)
    #and this one for small jets
    #lnRDR = np.log(0.4/all_lund_drs)
    
    path_to_save = config['data']['path_to_save']
    
    #Time for the plotting!!!!!
    #dr vs 1/z here
    plt.figure()
    plt.hist2d(lnRDR, ln1z, bins = 40, range = [[0,5],[0,5]], cmap = 'viridis',vmin=1)
    plt.colorbar()
    plt.xlabel("ln(R/dR)")
    plt.ylabel("ln(1/z)")

    filename_lund_z = path_to_save + "zlundplane"
    plt.savefig(str(filename_lund_z))
    print("1/z lund plane generated!")
    #dr vs kt here
    plt.figure()
    plt.hist2d(lnRDR, lnkt, bins = 40, range = [[0,5],[0,5]], cmap = 'viridis',vmin=1)
    plt.colorbar()
    plt.xlabel("ln(R/dR)")
    plt.ylabel("ln(kT)")

    filename_lund_kt = path_to_save + "ktlundplane"
    plt.savefig(str(filename_lund_kt))
    print("kt lund plane generated!")