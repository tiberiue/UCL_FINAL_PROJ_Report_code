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
import yaml
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataListLoader, DataLoader
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv, global_mean_pool, DataParallel, EdgeConv, GATConv,PNAConv
from torch_geometric.data import Data
import scipy.sparse as ss
from datetime import datetime, timedelta
from torch_geometric.utils import degree
import os.path

from tools.GNN_model_weight.models import *
from tools.GNN_model_weight.utils  import *

print("Libraries loaded!")

scale_factor = 14.475606  

for variation in range(31, 32):
    ## top tagging
    if variation==30:
        path_to_test_file = "/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/Testfile/user.teris.small.zprime*.tree.root_test.root"
        path_to_outdir = "/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/weightscoreZ/"
        path_to_combined_ckpt = "/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/weightlundZ/LundNet_weighted_Z_properbackg_e060_4115.52310.pt"
        output_name = "LundNetScores_combined"
        choose_model = "LundNet"
        learning_rate = 0.0005
        batch_size = 2048
        scale_factor = 1

    if variation==31:
        path_to_test_file = "/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/Testfile/user.teris.small.zprime*.tree.root_test.root"
        path_to_outdir = "/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/weightscoreZ/"
        path_to_combined_ckpt = "/eos/user/t/tmlinare/Lund_tagger/ljptagger/Models/weightlundZ/LundNet_weighted_Z_properbackg_e060_4115.52310.pt"
        output_name = "LundNetScores_combined"
        choose_model = "LundNet"
        learning_rate = 0.0005
        batch_size = 2048
        scale_factor = 1


    dsids = np.array([])
    mcweights = np.array([])
    ptweights = np.array([])
            
    NBHadrons = np.array([])
    parent1 =  np.array([])
    parent2 = np.array([])

            ## truth jet pt 
            # truth_jetpt = tree["Truth_jetPt"].array(library="np")
            # truth_ungroomedjet_pt = tree["Akt10UFOJet_ungroomed_truthJet_pt"].array(library="np")
            # truth_ungroomedjet_m = tree["Akt10UFOJet_ungroomed_truthJet_m"].array(library="np")
            # truth_ungroomedjet_split12 = tree["Akt10UFOJet_ungroomed_truthJet_Split12"].array(library="np") 
            
    jet_pts = np.array([])
    jet_etas = np.array([])
    jet_phis = np.array([])
    jet_ms =  np.array([])
            
    all_lund_zs = np.array([])
    all_lund_kts =  np.array([])
    all_lund_drs = np.array([])
    N_tracks = np.array([])
    
    files = glob.glob(path_to_test_file)

    #print ("files:",files)
    intreename = "FlatSubstructureJetTree"

    nentries_total = 0
    nentries_done = 0

    for file in files:
        nentries_total += uproot3.numentries(file, intreename)

    print("Evaluating on {} files with {} entries in total.".format(len(files), nentries_total))
    
    #Load tf keras model
    # jet_type = "Akt10RecoChargedJet" #track jets
    jet_type = "Akt10UFOJet" #UFO jets

    t_filestart = time.time()

    for file in files:
        t_start = time.time()
        dataset = []
        print("Loading file",file)

        with uproot.open(file) as infile:
            tree = infile[intreename]
            dsids = np.append(dsids,tree["DSID"].array())
            mcweights = np.append(mcweights,tree["mcWeight"].array())
            ptweights = np.append(ptweights,tree["fjet_testing_weight_pt"].array())
            
            NBHadrons = np.append(NBHadrons, tree["Akt10UFOJet_GhostBHadronsFinalCount"].array())
            parent1 =  np.append(parent1,tree["UFO_edge1"].array(library="np"),axis = 0)
            parent2 = np.append(parent2,tree["UFO_edge2"].array(library="np"),axis = 0)

            ## truth jet pt 
            # truth_jetpt = tree["Truth_jetPt"].array(library="np")
            # truth_ungroomedjet_pt = tree["Akt10UFOJet_ungroomed_truthJet_pt"].array(library="np")
            # truth_ungroomedjet_m = tree["Akt10UFOJet_ungroomed_truthJet_m"].array(library="np")
            # truth_ungroomedjet_split12 = tree["Akt10UFOJet_ungroomed_truthJet_Split12"].array(library="np") 
            
            jet_pts = np.append(jet_pts,tree["UFOSD_jetPt"].array(library="np"))
            jet_etas = np.append(jet_etas,tree["UFOSD_jetEta"].array(library="np"))
            jet_phis = np.append(jet_phis,tree["UFOSD_jetPhi"].array(library="np"))
            jet_ms =  np.append(jet_ms,tree["UFOSD_jetM"].array(library = "np") )
            
            all_lund_zs = np.append(all_lund_zs,tree["UFO_jetLundz"].array(library="np"))
            all_lund_kts =  np.append(all_lund_kts,tree["UFO_jetLundKt"].array(library="np"))
            all_lund_drs = np.append(all_lund_drs,tree["UFO_jetLundDeltaR"].array(library="np"))
            N_tracks = np.append(N_tracks,tree["UFO_Ntrk"].array(library="np"))
       
    # labels = ( dsids > 370000 ) & ( NBHadrons == 0 ) ## W tagging
    labels = ( dsids > 370000 ) & ( NBHadrons >= 1 ) ## top tagging
    labels = to_categorical(labels, 2)
    labels = np.reshape(labels[:,1], (len(all_lund_zs), 1))
    #flat_weights = GetPtWeight_2( dsids, jet_pts, SF=scale_factor)
    dataset = create_train_dataset_fulld_new_Ntrk_pt_weight_file_test( dataset , all_lund_zs, all_lund_kts, all_lund_drs, parent1, parent2, labels ,N_tracks,jet_pts, jet_ms  )
    # dataset = create_dataset_test( dataset , all_lund_zs, all_lund_kts, all_lund_drs, parent1, parent2, labels ,N_tracks,jet_pts, jet_ms  )
     
    #######################################################################################################################
    s_evt = 0
    events = 100
    print("Dataset created!")
    delta_t_fileax = time.time() - t_start
    print("Created dataset in {:.4f} seconds.".format(delta_t_fileax))

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print ("dataset dataset size:", len(dataset))


    #EVALUATING
    #torch.save(model.state_dict(), path)

    if choose_model == "LundNet":
        model = LundNet()
        # model = LundNet_old()
    if choose_model == "GATNet":
        model = GATNet()
    if choose_model == "GINNet":
        model = GINNet()
    if choose_model == "EdgeGinNet":
        model = EdgeGinNet()
    if choose_model == "PNANet":
        model = PNANet()

    model.load_state_dict(torch.load(path_to_combined_ckpt))

    device = torch.device('cuda') # Usually gpu 4 worked best, it had the most memory available
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #Predict scores
    y_pred = get_scores(test_loader, model, device)
    #print(y_pred)
    tagger_scores = y_pred[:,0]
    delta_t_pred = time.time() - t_filestart - delta_t_fileax
    print("Calculated predicitions in {:.4f} seconds,".format(delta_t_pred))
        
    #Save root files containing model scores
    filename = file.split("/")[-1]
    outfile_path = os.path.join(path_to_outdir, filename)

    # tagger_scores += [-1] * (len(dsids) - len(tagger_scores))
    tagger_scores = np.array(tagger_scores)
    # tagger_scores = np.pad(tagger_scores, (0, len(dsids) - len(tagger_scores)), 'constant', constant_values=(-1))
    print ("dsids",len(dsids),"mcweights",len(mcweights),"NBHadrons",len(NBHadrons),"tagger_scores",len(tagger_scores),"jet_pts",len(jet_pts),"jet_etas",len(jet_phis),"jet_phis",len(jet_phis),"jet_ms",len(jet_ms),"ptweights",len(ptweights))
    with uproot3.recreate("{}_score_{}.root".format(outfile_path, output_name)) as f:
        treename = "FlatSubstructureJetTree"
        #Declare branch data types
        f[treename] = uproot3.newtree({"EventInfo_mcChannelNumber": "int32",
                                        "EventInfo_mcEventWeight": "float32",
                                        "EventInfo_NBHadrons": "int32",   # I doubt saving the parents is necessary here
                                        "fjet_nnscore": "float32",        # which is why I didn't include them
                                        "fjet_pt": "float32",
                                        "fjet_eta": "float32",
                                        "fjet_phi": "float32",
                                        "fjet_m": "float32",
                                        "fjet_weight_pt": "float32", 
                                        # "truthjet_pt" : "float32",
                                        # "ungroomedtruthjet_pt" : "float32",
                                        # "ungroomedtruthjet_m" : "float32",
                                        # "ungroomedtruthjet_split12" : "float32",
                                        })

    
        #Save branches
        f[treename].extend({"EventInfo_mcChannelNumber": dsids,
                            "EventInfo_mcEventWeight": mcweights,
                            "EventInfo_NBHadrons": NBHadrons,
                            "fjet_nnscore": tagger_scores,
                            "fjet_pt": jet_pts,
                            "fjet_eta": jet_etas,
                            "fjet_phi": jet_phis,
                            "fjet_m": jet_ms,
                            "fjet_weight_pt": ptweights,
                            # "truthjet_pt" : truth_jetpt,
                            # "ungroomedtruthjet_pt" : truth_ungroomedjet_pt,
                            # "ungroomedtruthjet_m" : truth_ungroomedjet_m,
                            # "ungroomedtruthjet_split12" : truth_ungroomedjet_split12,
                            })

    delta_t_save = time.time() - t_start - delta_t_fileax - delta_t_pred
    print("Saved data in {:.4f} seconds.".format(delta_t_save))

    #nentries = 0
    #Time statistics
    nentries_done += uproot3.numentries(file, intreename)
    time_per_entry = (time.time() - t_start)/nentries_done
    eta = time_per_entry * (nentries_total - nentries_done)

    print("Evaluated on {} out of {} events".format(nentries_done, nentries_total))
    print("Estimated time until completion: {}".format(str(timedelta(seconds=eta))))


    print("Total evaluation time: {:.4f} seconds.".format(time.time()-t_filestart))