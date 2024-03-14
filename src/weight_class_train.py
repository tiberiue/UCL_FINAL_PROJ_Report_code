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
from datetime import datetime, timedelta
from torch_geometric.utils import degree
from torch_geometric.data import DataListLoader, DataLoader

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd

from tools.GNN_model_weight.models import *
from tools.GNN_model_weight.utils  import *

print("Libraries loaded!")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train with configurations')
    add_arg = parser.add_argument
    add_arg('config', help="job configuration")
    args = parser.parse_args()
    config_file = args.config
    config = load_yaml(config_file)

    path_to_file = config['data']['path_to_trainfiles']
    files = glob.glob(path_to_file)
    #files = glob.glob("/sps/atlas/k/khandoga/MySamplesS40/user.rvinasco.27045978._000004.tree.root_train.root")
    #files = files[:1]
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
    jet_eta = np.array([])
    jet_phi = np.array([])
    eta = np.array([])
    jet_truth_pts = np.array([])
    jet_truth_etas = np.array([])
    jet_truth_dRmatched = np.array([])
    jet_truth_split = np.array([])
    jet_ungroomed_ms = np.array([])
    jet_ungroomed_pts = np.array([])
    vector = []
    dataset = []

    for file in files:
        print("Loading file",file)
        with uproot.open(file) as infile:
            tree = infile[intreename]

            method = 1
            choose_model = config['architecture']['choose_model']
            
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
                jet_eta = tree["UFOSD_jetEta"].array(library="np")
                jet_phi = tree["UFOSD_jetPhi"].array(library="np")

                #print(all_lund_drs.shape ,"drs array size and amount of jets in total")
                #print(jet_pts[NBHadrons < 1 ].shape)
                
                #if dsids[0]>370000:
                    
                    #dsids = np.delete(dsids,np.argwhere(NBHadrons < 1))
                    #print(dsids.shape)
                    #parent1 = np.delete(parent1,np.argwhere(NBHadrons < 1),axis = 0)
                    #print(parent1.shape)
                    #parent2 = np.delete(parent2,np.argwhere(NBHadrons < 1),axis = 0)
                    #print(parent2.shape)
                    #jet_ms = np.delete(jet_ms,np.argwhere(NBHadrons < 1),axis = 0)
                    #print(jet_ms.shape)
                    #N_tracks = np.delete(N_tracks,np.argwhere(NBHadrons < 1),axis = 0)
                    #print(N_tracks.shape)
                    #jet_pts = np.delete(jet_pts,np.argwhere(NBHadrons < 1),axis = 0)
                    #print(jet_pts.shape)
                    #jet_eta = np.delete(jet_eta,np.argwhere(NBHadrons < 1),axis = 0)
                    #jet_phi = np.delete(jet_phi,np.argwhere(NBHadrons < 1),axis = 0)
                    #all_lund_zs  = np.delete(all_lund_zs,np.argwhere(NBHadrons < 1),axis = 0)
                    #print(all_lund_zs.shape)
                    #all_lund_kts = np.delete(all_lund_kts,np.argwhere(NBHadrons < 1),axis = 0)
                    #all_lund_drs = np.delete(all_lund_drs,np.argwhere(NBHadrons < 1),axis = 0)
                    #NBHadrons = np.delete(NBHadrons,np.argwhere(NBHadrons < 1))
                    #print(NBHadrons.shape)
                #W boson labeling
                labels = ( dsids > 370000 ) & ( NBHadrons == 0 )
                #Z prime boson labeling
                #labels = ( dsids > 370000 ) & ( NBHadrons >= 1 )
        
                labels = to_categorical(labels, 2)
                labels = np.reshape(labels[:,1], (len(all_lund_zs), 1))

                print (int(labels.sum()),"labeled as signal out of", len(labels), "total events")
                #using weight file
                #flat_weights = GetPtWeight_2( dsids, jet_pts, filename=config['data']['weights_file'], SF=config['data']['scale_factor'])
                #using utils weight measuring
                flat_weights = GetPtWeight_2( dsids, jet_pts, filename=config['data']['weights_file'],SF=config['data']['scale_factor'])
                #dataset = create_train_dataset_fulld_new_Ntrk_pt_weight_file( dataset , all_lund_zs, all_lund_kts, all_lund_drs, parent1, parent2, flat_weights, labels ,N_tracks, jet_pts , jet_ms)
                print("Flat weights collected")
                if choose_model=='LundNet_Ntrk_Plus':
                    Tau21 = tree["UFO_Tau12_wta"].array(library="np")
                    #Tau21 = np.log( tree["UFO_Tau12_wta"].array(library="np") / 10 )
                    C2 = tree["UFO_C2"].array(library="np") * 10
                    D2 = tree["UFO_D2"].array(library="np") 
                    Angularity = tree["UFO_Angularity"].array(library="np") * 1000
                    FoxWolfram20 = tree["UFO_FoxWolfram20"].array(library="np")
                    KtDR = tree["UFO_KtDR"].array(library="np")
                    PlanarFlow = tree["UFO_PlanarFlow"].array(library="np")
                    Split12 = tree["Akt10UFOJet_Split12"].array(library="np") / 10000
                    ZCut12 = tree["UFO_ZCut12"].array(library="np")
                    
                    print("----------Tau21", np.amax(Tau21),"  ", np.amin(Tau21) )
                    print("----------C2",np.amax(C2),"  ", np.amin(C2) )
                    print("----------D2",np.amax(D2),"  ", np.amin(D2) )
                    print("----------Angularity",np.amax(Angularity),"  ", np.amin(Angularity) )
                    print("----------FoxWolfram20",np.amax(FoxWolfram20),"  ", np.amin(FoxWolfram20) )
                    print("----------KtDR",np.amax(KtDR),"  ", np.amin(KtDR) )
                    print("----------PlanarFlow",np.amax(PlanarFlow),"  ", np.amin(PlanarFlow) )
                    print("----------Split12",np.amax(Split12),"  ", np.amin(Split12) )
                    print("----------ZCut12",np.amax(ZCut12),"  ", np.amin(ZCut12) )
                    
                    dataset = create_train_dataset_fulld_new_Ntrk_pt_weight_file_PLUS( dataset , all_lund_zs, all_lund_kts, all_lund_drs, parent1, parent2, flat_weights, labels ,N_tracks, jet_pts , jet_ms, Tau21, C2, D2, Angularity, FoxWolfram20, KtDR, PlanarFlow, Split12, ZCut12)
                else:
                    dataset = create_train_dataset_fulld_new_Ntrk_pt_weight_file( dataset , all_lund_zs, all_lund_kts, all_lund_drs, parent1, parent2, flat_weights, labels ,N_tracks, jet_pts , jet_ms)
                    #gc.collect()
                    
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


        
    
    print("Dataset created!")
    delta_t_fileax = time.time() - t_start
    print("Created dataset in {:.4f} seconds.".format(delta_t_fileax))

    ## define architecture
    batch_size = config['architecture']['batch_size']
    test_size = config['architecture']['test_size']


    dataset= shuffle(dataset,random_state=42)
    train_ds, validation_ds = train_test_split(dataset, test_size = test_size, random_state = 144)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_ds, batch_size=batch_size, shuffle=False)

    delta_t_fileax = time.time() - t_start
    print("Splitted datasets in {:.4f} seconds.".format(delta_t_fileax))


    print ("train dataset size:", len(train_ds))
    print ("validation dataset size:", len(validation_ds))
    
    
    deg = torch.zeros(10, dtype=torch.long)
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    

    n_epochs = config['architecture']['n_epochs']
    learning_rate = config['architecture']['learning_rate']
    choose_model = config['architecture']['choose_model']
    save_every_epoch = config['architecture']['save_every_epoch']

    if choose_model == "LundNet":
        model = LundNet()
    if choose_model == "GATNet":
        model = GATNet()
    if choose_model == "GINNet":
        model = GINNet()
    if choose_model == "EdgeGinNet":
        model = EdgeGinNet()
    if choose_model == "PNANet":
        model = PNANet()
    if choose_model == "LundNet_Ntrk_Plus":
        model = LundNet_Ntrk_Plus() 


    flag = config['retrain']['flag']
    path_to_ckpt = config['retrain']['path_to_ckpt']

    if flag==True:
        path = path_to_ckpt
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        #model.load_state_dict(torch.load(path))

    #device = torch.device('cpu')
    device = torch.device('cuda') # Usually gpu 4 worked best, it had the most memory available
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    train_jds = []
    val_jds = []

    train_bgrej = []
    val_bgrej = []

    model_name = config['data']['model_name']
    path_to_save = config['data']['path_to_save']
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    metrics_filename = path_to_save+"losses_"+model_name+datetime.now().strftime("%d%m-%H%M")+".txt"

    for epoch in range(n_epochs):
        print("Epoch:{}".format(epoch+1))
        train_loss.append(train(train_loader, model, device, optimizer))
    #    print ("Train Loss:",train_loss[-1])
    #    delta_t_fileax = time.time() - t_start
    #    print("trained epoch in {:.4f} seconds.".format(delta_t_fileax))
        val_loss.append(my_test(val_loader, model, device))

        #epsilon_bg, jds = aux_metrics(train_loader)
        epsilon_bg, jds = 0,0
        train_jds.append(jds)
        print ("Train epsilon bg:",epsilon_bg)
        print ("Train JDV:",jds)
        train_bgrej.append(epsilon_bg)

        #epsilon_bg_test, jds_test = aux_metrics(val_loader)
        epsilon_bg_test, jds_test = 0,0
        val_jds.append(jds_test)
        val_bgrej.append(epsilon_bg_test)
        print ("Test epsilon bg:",epsilon_bg_test)
        print ("Test JDV:",jds_test)

        print('Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f},train_jds: {:.5f},val_jds: {:.5f}'.format(epoch, train_loss[epoch], val_loss[epoch], train_jds[epoch], val_jds[epoch]))
        metrics = pd.DataFrame({"Train_Loss":train_loss,"Val_Loss":val_loss, "Train_jds":train_jds,"Val_jds":val_jds,"Train_bgrej":train_bgrej,"Val_bgrej":val_bgrej})

        metrics.to_csv(metrics_filename, index = False)

    #    print('Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(epoch+1, train_loss[epoch], val_loss[epoch]))
        if (save_every_epoch):
            torch.save(model.state_dict(), path_to_save+model_name+"e{:03d}".format(epoch+1)+"_{:.5f}".format(val_loss[epoch])+".pt")
        elif epoch == n_epochs-1:
            torch.save(model.state_dict(), path_to_save+model_name+"e{:03d}".format(epoch+1)+"_{:.5f}".format(val_loss[epoch])+".pt")
