import argparse
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
import yaml
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

from tools.GNN_model_weight.models import *
from tools.GNN_model_weight.utils  import *

loss_weights = [1,1]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train with configurations')
    add_arg = parser.add_argument
    add_arg('config', help="job configuration")
    args = parser.parse_args()
    config_file = args.config
    config = load_yaml(config_file)


    path_to_file = config['data']['path_to_trainfiles']
    files = glob.glob(path_to_file)

    jet_type = "Akt10UFOJet" #UFO jets
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

    jet_pts_adv = np.array([])
    jet_ms_adv = np.array([])
    vector = []
    dataset = []


    for file in files:
        print("Loading file",file)
        with uproot.open(file) as infile:
            tree = infile[intreename]

            method = 1
            if method==0 :
                dsids = np.append( dsids, np.array(tree["DSID"].array()) )
                #eta = ak.concatenate(eta, pad_ak3(tree["Akt10TruthJet_jetEta"].array(), 30),axis=0)                                       
                mcweights = tree["mcWeight"].array()
                NBHadrons = np.append( NBHadrons, ak.to_numpy(tree["Akt10UFOJet_GhostBHadronsFinalCount"].array()))

                parent1 = np.append(parent1, tree["UFO_edge1"].array(library="np"),axis=0)
                parent2 = np.append(parent2, tree["UFO_edge2"].array(library="np"),axis=0)
                jet_ms = np.append(jet_ms, ak.to_numpy(tree["UFOSD_jetM"].array()))
                jet_pts = np.append(jet_pts, ak.to_numpy(tree["UFOSD_jetPt"].array()))

                #Get jet kinematics                                                                                           

                #Get Lund variables                                                                                
                all_lund_zs = np.append(all_lund_zs,tree["UFO_jetLundz"].array(library="np") )
                all_lund_kts = np.append(all_lund_kts, tree["UFO_jetLundKt"].array(library="np") )
                all_lund_drs = np.append(all_lund_drs, tree["UFO_jetLundDeltaR"].array(library="np") )

            if method==1 :
                #jet_pts_adv = np.append(jet_pts_adv, ak.to_numpy(tree["UFOSD_jetPt"].array()))
                #jet_ms_adv = np.append(jet_ms_adv, ak.to_numpy(tree["UFOSD_jetM"].array()))
                
                dsids = tree["DSID"].array(library="np")
                NBHadrons = tree["Akt10UFOJet_GhostBHadronsFinalCount"].array(library="np")
                parent1 =  tree["UFO_edge1"].array(library="np")
                parent2 = tree["UFO_edge2"].array(library="np")
                jet_ms =  ak.to_numpy(tree["UFOSD_jetM"].array() )
                all_lund_zs = tree["UFO_jetLundz"].array(library="np")
                all_lund_kts =  tree["UFO_jetLundKt"].array(library="np")
                all_lund_drs = tree["UFO_jetLundDeltaR"].array(library="np")
                N_tracks = tree["UFO_Ntrk"].array(library="np")
                jet_pts = tree["UFOSD_jetPt"].array(library="np")
                labels = ( dsids > 370000 ) & ( NBHadrons == 0 )
                labels = to_categorical(labels, 2)
                labels = np.reshape(labels[:,1], (len(all_lund_zs), 1))
                flat_weights = GetPtWeight_2( dsids, jet_pts, filename=config['data']['weights_file'], SF=config['data']['scale_factor'])
                dataset = create_train_dataset_fulld_new_Ntrk_pt_weight_file( dataset , all_lund_zs, all_lund_kts, all_lund_drs, parent1, parent2, flat_weights, labels ,N_tracks, jet_pts, jet_ms )

    if method==0 :
        #Get labels                                                                                                                                                       
        labels = ( dsids > 370000 ) & ( NBHadrons == 0 )
        flat_weights = np.vectorize(GetPtWeight)(dsids, jet_pts, filename=config['data']['weights_file'], SF=config['data']['scale_factor'])
        #print(labels)                                                                                                                                             
        labels = to_categorical(labels, 2)
        labels = np.reshape(labels[:,1], (len(all_lund_zs), 1))

        print (int(labels.sum()),"labeled as signal out of", len(labels), "total events")

        delta_t_fileax = time.time() - t_start
        print("Opened data in {:.4f} seconds.".format(delta_t_fileax))

        dataset = create_train_dataset_fulld_new(all_lund_zs, all_lund_kts, all_lund_drs, parent1, parent2, flat_weights, labels) ## add weights                                                       
        #dataset = create_train_dataset_fulld_new(all_lund_zs[s_evt:events], all_lund_kts[s_evt:events], all_lund_drs[s_evt:events], parent1[s_evt:events], parent2[s_evt:events], labels[s_evt:events]) 
        
        

    deg = torch.zeros(10, dtype=torch.long)
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())


    num_gaussians = config["architecture"]["num_gaussians"] # number of Gaussians at the end


    #ms = np.array(jet_ms).reshape(len(jet_ms), 1)
    #pts = np.array(np.log(jet_pts)).reshape(len(jet_pts), 1)
    #ms = np.array(jet_ms_adv).reshape(len(jet_ms_adv), 1)
    #pts = np.array(np.log(jet_pts_adv)).reshape(len(jet_pts_adv), 1)
    
    batch_size = config["architecture"]["batch_size"]
    test_size = config['architecture']['test_size']
    learning_rate = config['architecture']['learning_rate']

    #test_ds, test1_ds = train_test_split(dataset, test_size = test_size, random_state = 144)
    #test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    
    ##################3###
    #adv_dataset = create_adversary_trainset(pts, ms)
    #conc_dataset = ConcatDataset(dataset, adv_dataset)


    #conc_dataset= shuffle(conc_dataset,random_state=0)
    #train_ds, validation_ds = train_test_split(conc_dataset, test_size = test_size, random_state = 144)
    train_ds, validation_ds = train_test_split(dataset, test_size = test_size, random_state = 144)


    adv_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_ds, batch_size=batch_size, shuffle=False)

    print ("train dataset size:", len(train_ds))
    print ("validation dataset size:", len(validation_ds))


    print ("Loading classifier model.")
    
    
    path_to_classifier_ckpt = config['classifier']['path_to_classifier_ckpt']
    choose_model = config['classifier']['choose_model']
    
    if choose_model == "LundNet":
        clsf = LundNet()
    if choose_model == "GATNet":
        clsf = GATNet()
    if choose_model == "GINNet":
        clsf = GINNet()
    if choose_model == "EdgeGinNet":
        clsf = EdgeGinNet()
    if choose_model == "PNANet":
        clsf = PNANet()
    
    clsf.load_state_dict(torch.load(path_to_classifier_ckpt))
    #clsf.load_state_dict(torch.load(path_to_classifier_ckpt, map_location=torch.device('cpu')))
    
    print ("Classifier model loaded, training adversary.")

    lambda_parameter = config["architecture"]["lambda_parameter"]
    loss_parameter = config["architecture"]["loss_parameter"]
    

    #adv = Adversary(lambda_parameter, num_gaussians)
    adv = Adversary_new(lambda_parameter, num_gaussians)
    
    #adv_model_weights = "/sps/atlas/k/khandoga/TrainGNN/Models/adv_e003_5.40484.pt"
    #adv.load_state_dict(torch.load(adv_model_weights))


    for param in clsf.parameters():
        param.require_grads = False

    
    device = torch.device('cuda') # Usually gpu 4 worked best, it had the most memory available
    #device = torch.device('cpu')
    
    clsf.to(device)
    adv.to(device)
    optimizer = torch.optim.Adam(adv.parameters(), lr=learning_rate)
    #optimizer = torch.optim.Adam(list(clsf.parameters()) + list(adv.parameters()), lr=0.0005)


    print("Training adversary whilst keeping classifier the same.")

    train_loss_clsf = []
    train_loss_adv = []
    train_loss_total = []

    val_loss_clsf = []
    val_loss_adv = []
    val_loss_total = []

    train_acc = []
    val_acc = []
    path_to_store = config['adversary']['path_to_store'] ## path to store models txt files
    
    save_adv_every_epoch = config['adversary']['save_adv_every_epoch']
    adv_model_name = config['adversary']['adv_model_name']
    n_epochs_adv = config['adversary']['n_epochs_adv']
    metrics_filename = path_to_store+"losses_"+adv_model_name+datetime.now().strftime("%d%m-%H%M")+".txt"


    for epoch in range(n_epochs_adv): # this may need to be bigger
     #   my_test(test_loader)

        ad_lt, clsf_lt, total_lt =  train_adversary_2(adv_loader, clsf, adv, optimizer, device, loss_parameter ,loss_weights) ## loss = loss1 - loss_parameter*loss2
        train_loss_adv.append(ad_lt)
        train_loss_clsf.append(clsf_lt)
        train_loss_total.append(total_lt)
        train_acc.append(get_accuracy(adv_loader, clsf, device))

        ad_lv, clsf_lv, total_lv =  test_combined(val_loader, clsf, adv, device, loss_parameter , loss_weights) ## loss = loss1 - loss_parameter*loss2
        val_loss_adv.append(ad_lv)
        val_loss_clsf.append(clsf_lv)
        val_loss_total.append(total_lv)
        val_acc.append(get_accuracy(val_loader, clsf, device))

        print('Epoch: {:03d}, Train Loss total: {:.5f}, Train Loss adv: {:.5f}, Train Loss clsf: {:.5f}, val_loss_adv: {:.5f}, val_loss_clsf: {:.5f}, val_loss_total: {:.5f},train_acc: {:.5f},val_acc: {:.5f}'.format(epoch, train_loss_total[epoch],train_loss_adv[epoch],train_loss_clsf[epoch], val_loss_adv[epoch], val_loss_clsf[epoch], val_loss_total[epoch],train_acc[epoch],val_acc[epoch]))
        metrics = pd.DataFrame({"Train_Loss_adv":train_loss_adv,"Train_Loss_clsf":train_loss_clsf,"Train_Loss_total":train_loss_total,"Val_Loss_Adv":val_loss_adv,"Val_loss_Class":val_loss_clsf,"val_loss_total":val_loss_total, "Train_Acc":train_acc,"Val_Acc":val_acc})
        metrics.to_csv(metrics_filename, index = False)
        if (save_adv_every_epoch):
            torch.save(adv.state_dict(), path_to_store+adv_model_name+"e{:03d}".format(epoch+1)+"_{:.5f}".format(val_loss_adv[epoch])+".pt")
        elif epoch == n_epochs-1:
            torch.save(adv.state_dict(), path_to_store+adv_model_name+"e{:03d}".format(epoch+1)+"_{:.5f}".format(val_loss_adv[epoch])+".pt")
