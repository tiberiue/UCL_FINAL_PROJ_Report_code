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

    path_to_test_file = config['data']['path_to_test_file']
    files = glob.glob(path_to_test_file)

    print ("files:",files)
    intreename = "FlatSubstructureJetTree"
    path_to_outdir = config['data']['path_to_outdir']


    path_to_combined_ckpt = config['test']['path_to_combined_ckpt']

    output_name = config['test']['output_name']

    nentries_total = 0
    nentries_done = 0

    for file in files:
        nentries_total += uproot3.numentries(file, intreename)

    print("Evaluating on {} files with {} entries in total.".format(len(files), nentries_total))


    #Load tf keras model
    # jet_type = "Akt10RecoChargedJet" #track jets
    jet_type = "Akt10UFOJet" #UFO jets
    #print(files)

    t_filestart = time.time()

    for file in files:

        t_start = time.time()

        dsids = np.array([])
        NBHadrons = np.array([])
        mcweights = np.array([])
        ptweights = np.array([])

        all_lund_zs = np.array([])
        all_lund_kts = np.array([])
        all_lund_drs = np.array([])

        parent1 = np.array([])
        parent2 = np.array([])

        jet_pts = np.array([])
        jet_phis = np.array([])
        jet_etas = np.array([])
        jet_ms = np.array([])
        eta = np.array([])
        jet_truth_pts = np.array([])
        jet_truth_etas = np.array([])
        jet_truth_dRmatched = np.array([])
        jet_truth_split = np.array([])
        jet_ungroomed_ms = np.array([])
        jet_ungroomed_pts = np.array([])
        vector = []

        print("Loading file",file)

        #with uproot.open(file) as infile:
        with uproot.open(file) as infile:

            tree = infile[intreename]
            dsids = np.append( dsids, np.array(tree["DSID"].array()) )
            #eta = ak.concatenate(eta, pad_ak3(tree["Akt10TruthJet_jetEta"].array(), 30),axis=0)
            mcweights = np.append( mcweights, np.array(tree["mcWeight"].array()) )
            ptweights = np.append( ptweights, np.array(tree["fjet_testing_weight_pt"].array()) )
            NBHadrons = np.append( NBHadrons, ak.to_numpy(tree["Akt10UFOJet_GhostBHadronsFinalCount"].array()))

            parent1 = np.append(parent1, tree["UFO_edge1"].array(library="np"),axis=0)
            parent2 = np.append(parent2, tree["UFO_edge2"].array(library="np"),axis=0)

            #Get jet kinematics
            jet_pts = np.append(jet_pts, tree["UFOSD_jetPt"].array(library="np"))
            jet_etas = np.append(jet_etas, tree["UFOSD_jetEta"].array(library="np"))
            jet_phis = np.append(jet_phis, tree["UFOSD_jetPhi"].array(library="np"))
            jet_ms = np.append(jet_ms, tree["UFOSD_jetM"].array(library="np"))

    #        jet_truth_pts = np.append(jet_truth_pts, tree["Truth_jetPt"].array(library="np"))
    #        jet_truth_etas = np.append(jet_truth_etas, ak.to_numpy(tree["Truth_jetEta"].array(library="np")))


            #Get Lund variables
            all_lund_zs = np.append(all_lund_zs,tree["UFO_jetLundz"].array(library="np") )
            all_lund_kts = np.append(all_lund_kts, tree["UFO_jetLundKt"].array(library="np") )
            all_lund_drs = np.append(all_lund_drs, tree["UFO_jetLundDeltaR"].array(library="np") )



        #Get labels
        labels = ( dsids > 370000 ) & ( NBHadrons == 0 )
        labels = to_categorical(labels, 2)
        labels = np.reshape(labels[:,1], (len(all_lund_zs), 1))

        dataset = create_train_dataset_fulld_new(all_lund_zs, all_lund_kts, all_lund_drs, parent1, parent2, labels)
        s_evt = 0
        events = 100
        #dataset = create_train_dataset_fulld_new(all_lund_zs[s_evt:events], all_lund_kts[s_evt:events], all_lund_drs[s_evt:events], parent1[s_evt:events], parent2[s_evt:events], labels[s_evt:events])

        print("Dataset created!")
        delta_t_fileax = time.time() - t_start
        print("Created dataset in {:.4f} seconds.".format(delta_t_fileax))

        batch_size = config['test']['batch_size']

        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        print ("dataset dataset size:", len(dataset))


        #EVALUATING
        #torch.save(model.state_dict(), path)
        choose_model = config['test']['choose_model']
        learning_rate = config['test']['learning_rate']

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

        model.load_state_dict(torch.load(path_to_combined_ckpt))

        device = torch.device('cuda') # Usually gpu 4 worked best, it had the most memory available
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        #Predict scores
        y_pred = get_scores(test_loader, model, device)
        print(y_pred)
        tagger_scores = y_pred[:,0]
        delta_t_pred = time.time() - t_filestart - delta_t_fileax
        print("Calculated predicitions in {:.4f} seconds,".format(delta_t_pred))

        #Save root files containing model scores
        filename = file.split("/")[-1]
        outfile_path = os.path.join(path_to_outdir, filename)

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
                                          "fjet_weight_pt": "float32"
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
