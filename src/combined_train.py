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


from tools.GNN_model.models import *
from tools.GNN_model.utils  import *



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Combined train with configurations')
    add_arg = parser.add_argument
    add_arg('config', help="job configuration")
    args = parser.parse_args()
    config_file = args.config
    config = load_yaml(config_file)


    lr_ratio = config["combined"]["lr_ratio"]
    grad_parameter = config["combined"]["grad_parameter"]
    loss_parameter = config["combined"]["loss_parameter"]

    path_to_file = config['data']['path_to_trainfiles']
    files = glob.glob(path_to_file)    #files = files[:1]

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

    t_start = time.time()
    #W bosons
    dataset = create_train_dataset_fulld_new(all_lund_zs, all_lund_kts, all_lund_drs, parent1, parent2, labels)
    #train_loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    delta_t_fileax = time.time() - t_start
    print("Created dataset in {:.4f} seconds.".format(delta_t_fileax))


    deg = torch.zeros(10, dtype=torch.long)
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())



    num_gaussians = config["architecture"]["num_gaussians"] # number of Gaussians at the end
    batch_size = config["architecture"]["batch_size"]
    test_size = config["architecture"]["test_size"]


    ms = np.array(jet_ms).reshape(len(jet_ms), 1)
    pts = np.array(np.log(jet_pts)).reshape(len(jet_pts), 1)


    test_ds, test1_ds = train_test_split(dataset, test_size = test_size, random_state = 144)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    adv_dataset = create_adversary_trainset(pts, ms)
    conc_dataset = ConcatDataset(dataset, adv_dataset)


    conc_dataset= shuffle(conc_dataset,random_state=0)
    train_ds, validation_ds = train_test_split(conc_dataset, test_size = test_size, random_state = 144)


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

    print ("Classifier model loaded, loading adversary.")

    #adv = Adversary()
    adv = Adversary(grad_parameter, num_gaussians)
    path_to_adv_model_weights = config["adversary"]["path_to_adv_model_weights"]
    adv.load_state_dict(torch.load(path_to_adv_model_weights))
    print ("Adversary loaded.")

    lr_optimizer = config["combined"]["lr_optimizer"]

    device = torch.device('cuda') # Usually gpu 4 worked best, it had the most memory available
    clsf.to(device)
    adv.to(device)
    optimizer_cl = torch.optim.Adam(clsf.parameters(), lr=lr_optimizer)
    optimizer_adv = torch.optim.Adam(adv.parameters(), lr=lr_optimizer*lr_ratio)


    for param in clsf.parameters():
        param.require_grads = True



    MASSBINS = np.linspace(40, 300, (300 - 40) // 5 + 1, endpoint=True)





    print("Started training together!")

    train_loss_clsf = []
    train_loss_adv = []
    train_loss_total = []

    val_loss_clsf = []
    val_loss_adv = []
    val_loss_total = []

    train_acc = []
    val_acc = []

    train_jds = []
    val_jds = []

    train_bgrej = []
    val_bgrej = []

    train_jsdbg = []
    val_jsdbg = []

    path_to_save_combined = config["combined"]["path_to_save_combined"]## path to store models txt files
    model_name = f"class_grad{grad_parameter}_lossp{loss_parameter}_lrr{lr_ratio}_s50"
    metrics_filename = path_to_save_combined+"Models/"+"losses_"+model_name+datetime.now().strftime("%d%m-%H%M")+".txt"

    save_adv_every_epoch = config["combined"]["save_adv_every_epoch"]
    adv_model_name = f"adv_class_grad{grad_parameter}_lossp{loss_parameter}_lrr{lr_ratio}"

    n_epochs_common = config["combined"]["n_epochs_common"]
    save_every_epoch = config["combined"]["save_every_epoch"]

    #for param in adv.parameters():
    #    param.require_grads = True

    for epoch in range(n_epochs_common): # this may need to be bigger
        print("Epoch:{}".format(epoch))

        ad_lt, clsf_lt, total_lt =  train_combined(adv_loader, clsf, adv, optimizer_cl, optimizer_adv, device, loss_parameter)
        train_loss_clsf.append(clsf_lt)
        train_loss_adv.append(ad_lt)
        train_loss_total.append(total_lt)
        epsilon_bg, jds = aux_metrics(adv_loader, clsf, adv, device, MASSBINS)
        #epsilon_bg, jds = 0,0
        train_jds.append(jds)
        print ("Train epsilon bg:",epsilon_bg)
        print ("Train JDV:",jds)
        train_bgrej.append(epsilon_bg)
        if jds:
            train_jsdbg.append(epsilon_bg - 1/jds)
        else:
            train_jsdbg.append(0)

        ad_lv, clsf_lv, total_lv =  test_combined(val_loader, clsf, adv, device, loss_parameter) ## lambda_parameter is the factor here  loss = loss1 - lambda_parameter*loss2
        val_loss_adv.append(ad_lv)
        val_loss_clsf.append(clsf_lv)
        val_loss_total.append(total_lv)

        epsilon_bg_test, jds_test = aux_metrics(val_loader, clsf, adv, device, MASSBINS)
        #epsilon_bg_test, jds_test = 0,0
        val_jds.append(jds_test)
        val_bgrej.append(epsilon_bg_test)
        if jds_test:
            val_jsdbg.append(epsilon_bg_test - 1/jds_test)
        else:
            val_jsdbg.append(0)

        print ("Test epsilon bg:",epsilon_bg_test)
        print ("Test JDV:",jds_test)

        print('Epoch: {:03d}, Train Loss total: {:.5f}, Train Loss adv: {:.5f}, Train Loss clsf: {:.5f}, val_loss_adv: {:.5f}, val_loss_clsf: {:.5f}, val_loss_total: {:.5f},train_jds: {:.5f},val_jds: {:.5f},train_jdsbg: {:.5f},val_jdsbg: {:.5f}'.format(epoch,
            train_loss_total[epoch],train_loss_adv[epoch],train_loss_clsf[epoch], val_loss_adv[epoch], val_loss_clsf[epoch], val_loss_total[epoch], train_jds[epoch], val_jds[epoch],train_jsdbg[epoch],val_jsdbg[epoch]))
        metrics = pd.DataFrame({"Train_Loss_adv":train_loss_adv,"Train_Loss_clsf":train_loss_clsf,"Train_Loss_total":train_loss_total,"Val_Loss_Adv":val_loss_adv,"Val_loss_Class":val_loss_clsf,
            "val_loss_total":val_loss_total, "Train_jds":train_jds,"Val_jds":val_jds,"Train_bgrej":train_bgrej,"Val_bgrej":val_bgrej, "Train_jsdbg":train_jsdbg,"Val_jsdbg":val_jsdbg})
        metrics.to_csv(metrics_filename, index = False)
        if (save_every_epoch):
            torch.save(clsf.state_dict(), path_to_save_combined+"Models/"+model_name+"e{:03d}".format(epoch+1)+"_{:.5f}".format(val_loss_total[epoch])+"_comb_"+".pt")
            torch.save(adv.state_dict(), path_to_save_combined+"Models/"+adv_model_name+"e{:03d}".format(epoch+1)+"_{:.5f}".format(val_loss_adv[epoch])+"_comb_"+".pt")

      #  elif epoch == n_epochs-1:
      #      torch.save(clsf.state_dict(), path+"Models/"+model_name+"_ct_"+"e{:03d}".format(epoch+1)+"_{:.5f}".format(val_loss_total[epoch])+".pt")
      #      torch.save(adv.state_dict(), path+"Models/"+adv_model_name+"_ct_"+"e{:03d}".format(epoch+1)+"_{:.5f}".format(val_loss_adv[epoch])+".pt")
