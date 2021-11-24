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

from tools.GNN_model.models import *
from tools.GNN_model.utils  import *

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

            #Get jet kinematics

    #        jet_truth_split = np.append(jet_truth_split, tree["Akt10TruthJet_ungroomed_truthJet_Split12"].array(library="np"))

            #Get Lund variables
            all_lund_zs = np.append(all_lund_zs,tree["UFO_jetLundz"].array(library="np") )
            all_lund_kts = np.append(all_lund_kts, tree["UFO_jetLundKt"].array(library="np") )
            all_lund_drs = np.append(all_lund_drs, tree["UFO_jetLundDeltaR"].array(library="np") )



    #Get labels
    labels = ( dsids > 370000 ) & ( NBHadrons == 0 )



    #labels = (dsids > 370000) & (jet_truth_pts > 200) & (abs(jet_truth_etas) < 2) & \
    #(jet_ms > 40) & (jet_ms < 300)& (jet_pts > 200) & (jet_pts < 3000) & (jet_ungroomed_ms > 50000) & \
    #(NBHadrons == 0) & (abs(jet_truth_dRmatched) == 24) & \
    #(jet_truth_split/1000 > 55.25 * np.exp(-2.34/1000 * jet_ungroomed_pts/1000))

    #print(labels)
    labels = to_categorical(labels, 2)
    labels = np.reshape(labels[:,1], (len(all_lund_zs), 1))

    print (int(labels.sum()),"labeled as signal out of", len(labels), "total events")


    delta_t_fileax = time.time() - t_start
    print("Opened data in {:.4f} seconds.".format(delta_t_fileax))


    #W bosons
    # It will take about 30 minutes to finish
    dataset = create_train_dataset_fulld_new(all_lund_zs, all_lund_kts, all_lund_drs, parent1, parent2, labels)
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

    flag = config['retrain']['flag']
    path_to_ckpt = config['retrain']['path_to_ckpt']

    if flag==True:
        path = path_to_ckpt
        model.load_state_dict(torch.load(path))


    device = torch.device('cuda') # Usually gpu 4 worked best, it had the most memory available
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    train_jds = []
    val_jds = []

    train_bgrej = []
    val_bgrej = []

    model_name = "LundNet_longrun_"
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
