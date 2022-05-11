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
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataListLoader, DataLoader
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv, global_mean_pool, DataParallel, EdgeConv, GATConv,PNAConv
from torch_geometric.data import Data
import scipy.sparse as ss
from datetime import datetime, timedelta
import argparse
from torch_geometric.utils import degree
import os.path

print("Libraries loaded!")

#parser = argparse.ArgumentParser(description='Index helper')
#parser.add_argument('index', type=int, help='Beginning of slice')
#args = parser.parse_args()

#infiles_list = glob.glob("/eos/user/r/riordan/data/wprime/*_test.root") + glob.glob("/eos/user/r/riordan/data/j3to9/*_test.root")
#infiles_list = infiles_list[args.index:args.index+1]
#print(infiles_list)

#files = glob.glob("/eos/user/m/mykhando/public/FlatSamples/Train_Test_split_cclyon/user.mykhando.27245469._000004.tree.root_test.root")

files = glob.glob("/sps/atlas/k/khandoga/MySamplesS50/*test*.root")
#files = files[:1]
print ("files:",files)
intreename = "FlatSubstructureJetTree"

outdir = "/sps/atlas/k/khandoga/Scores/"

#path = "/sps/atlas/k/khandoga/TrainGNN/Models/LundNet_ufob_e010_0.12276.pt"
#path = "/sps/atlas/k/khandoga/TrainGNN/Models/class_neg_lambda10_jsd_loce037_-1.96575_comb_.pt"
#path = "/sps/atlas/k/khandoga/TrainGNN/Models/class_neg_grad05_jsd_longe112_-2.01732_comb_.pt"
path = "/sps/atlas/k/khandoga/TrainGNN/Models/classao_grad0.3_lossp0.04_lrr0.005_s50e020_-1.42204_comb_.pt"

#model_name = "LundTest_big"
model_name = "LundNet_weig_sc_0.08"    

    #input TTree name




nentries_total = 0
nentries_done = 0

for file in files:
#    if (os.path.isfile(outdir+"{}_score_{}.root".format(outfile_path, model_name))):
#        print ("Skipping file", outdir+"{}_score_{}.root".format(outfile_path, model_name))
#        continue 

    nentries_total += uproot3.numentries(file, intreename)

print("Evaluating on {} files with {} entries in total.".format(len(files), nentries_total))



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


   
@torch.no_grad()
def get_scores(loader):
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
        if i%10000 ==0:
            print ("Loading event:",i)
        if (len(edge1[i])== 0) or (len(edge2[i])== 0):
            edge = torch.tensor(np.array([edge1[i-1], edge2[i-1]]) , dtype=torch.long)
        else:
            edge = torch.tensor(np.array([edge1[i], edge2[i]]) , dtype=torch.long)
        vec = []
        vec.append(np.array([d[i], z[i], k[i]]).T)
        vec = np.array(vec)
        vec = np.squeeze(vec)
        graphs.append(Data(x=torch.tensor(vec, dtype=torch.float), edge_index=edge, y=torch.tensor(label[i], dtype=torch.float)))
    return graphs


#Load tf keras model
# jet_type = "Akt10RecoChargedJet" #track jets
jet_type = "Akt10UFOJet" #UFO jets
#print(files)
save_trained_model = True

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

    batch_size = 2048

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


    print ("dataset dataset size:", len(dataset))


    deg = torch.zeros(10, dtype=torch.long)
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())



    class PNANet(torch.nn.Module):
        def __init__(self,in_channels):
            aggregators = ['sum','mean', 'min', 'max', 'std']
            scalers = ['identity', 'amplification', 'attenuation',"linear",'inverse_linear']
            
            super(PNANet, self).__init__()
            self.conv1 = PNAConv(in_channels, out_channels=32, deg=deg, post_layers=3,aggregators=aggregators, 
                                                scalers = scalers)
            self.conv2 = PNAConv(in_channels=32, out_channels=64, deg=deg, post_layers=3,aggregators=aggregators, 
                                                scalers = scalers)
            self.conv3 = PNAConv(in_channels=64, out_channels=128, deg=deg, post_layers=3,aggregators=aggregators, 
                                                scalers = scalers)
            self.conv4 = PNAConv(in_channels=128, out_channels=256, deg=deg, post_layers=3,aggregators=aggregators, 
                                                scalers = scalers)
            self.conv5 = PNAConv(in_channels=256, out_channels=256, deg=deg, post_layers=3,aggregators=aggregators, 
                                                scalers = scalers)
            self.conv6 = PNAConv(in_channels=256, out_channels=256, deg=deg, post_layers=3,aggregators=aggregators, 
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



    #EVALUATING
    #torch.save(model.state_dict(), path)
    #model = PNANet(3)
    model = LundNet()
    model.load_state_dict(torch.load(path))
    device = torch.device('cuda') # Usually gpu 4 worked best, it had the most memory available
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    #Predict scores
    y_pred = get_scores(test_loader)
    print(y_pred)
    tagger_scores = y_pred[:,0]
    delta_t_pred = time.time() - t_filestart - delta_t_fileax
    print("Calculated predicitions in {:.4f} seconds,".format(delta_t_pred))

    #Save root files containing model scores
    filename = file.split("/")[-1]
    outfile_path = os.path.join(outdir, filename) 
        
    print ("dsids",len(dsids),"mcweights",len(mcweights),"NBHadrons",len(NBHadrons),"tagger_scores",len(tagger_scores),"jet_pts",len(jet_pts),"jet_etas",len(jet_phis),"jet_phis",len(jet_phis),"jet_ms",len(jet_ms),"ptweights",len(ptweights))
    with uproot3.recreate("{}_score_{}.root".format(outfile_path, model_name)) as f:

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

