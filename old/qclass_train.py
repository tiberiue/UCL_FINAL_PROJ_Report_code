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

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd



print("Libraries loaded!")

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

class GATNet(torch.nn.Module):
    def __init__(self, in_channels):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(in_channels, 16, heads=8, dropout=0.1)
        self.conv2 = GATConv(16 * 8, 16, heads=8, dropout=0.1)
        self.conv3 = GATConv(16 * 8, 32, heads=8, dropout=0.1)
        self.conv4 = GATConv(32 * 8, 32, heads=16, dropout=0.1)
        self.conv5 = GATConv(16 * 32, 64, heads=16, dropout=0.1)
        self.conv6 = GATConv(64 * 16, 64, heads=16, dropout=0.1)
        self.seq1 = nn.Sequential(nn.Linear(3072, 384), 
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

class GINNet(torch.nn.Module):
    def __init__(self):
        super(GINNet, self).__init__()
        self.conv1 = GINConv(nn.Sequential(nn.Linear(3, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU(), 
                                            nn.Linear(32, 32), 
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU()),train_eps=True)
        self.conv2 = GINConv(nn.Sequential(nn.Linear(32, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU(), 
                                            nn.Linear(32, 32),
                                            nn.BatchNorm1d(num_features=32),
                                            nn.ReLU()),train_eps=True)
        self.conv3 = GINConv(nn.Sequential(nn.Linear(32,64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU(), 
                                            nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU()),train_eps=True)
        self.conv4 = GINConv(nn.Sequential(nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU(), 
                                            nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU()),train_eps=True)
        self.conv5 = GINConv(nn.Sequential(nn.Linear(64, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU(), 
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU()),train_eps=True)
        self.conv6 = GINConv(nn.Sequential(nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU(), 
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU()),train_eps=True)
        
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

class EdgeGinNet(torch.nn.Module):
    def __init__(self):
        super(EdgeGinNet, self).__init__()
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
        self.conv4 = GINConv(nn.Sequential(nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU(), 
                                            nn.Linear(64, 64),
                                            nn.BatchNorm1d(num_features=64),
                                            nn.ReLU()),train_eps=True)
        self.conv5 = GINConv(nn.Sequential(nn.Linear(64, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU(), 
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU()),train_eps=True)
        self.conv6 = GINConv(nn.Sequential(nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU(), 
                                            nn.Linear(128, 128),
                                            nn.BatchNorm1d(num_features=128),
                                            nn.ReLU()),train_eps=True)
        
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



class PNANet(torch.nn.Module):
    def __init__(self,in_channels):
        aggregators = ['sum','mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation',"linear",'inverse_linear']
        
        super(PNANet, self).__init__()
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


files = glob.glob("/sps/atlas/k/khandoga/MySamplesS50/*_train.root")
#files = glob.glob("/sps/atlas/k/khandoga/MySamples90/user.mykhando.26845601._000005.tree.root_train.root")
#files = files[22:23]
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
intreename = "FlatSubstructureJetTree"
for file in files:

    print("Loading file",file)

    with uproot.open(file) as infile:
        tree = infile[intreename]
        dsids = np.append( dsids, np.array(tree["DSID"].array()) )
        #eta = ak.concatenate(eta, pad_ak3(tree["Akt10TruthJet_jetEta"].array(), 30),axis=0)
        mcweights = tree["mcWeight"].array()
        NBHadrons = np.append( NBHadrons, ak.to_numpy(tree["Akt10TruthJet_ungroomedParent_GhostBHadronsFinalCount"].array()))

        parent1 = np.append(parent1, tree["UFO_edge1"].array(library="np"),axis=0)
        parent2 = np.append(parent2, tree["UFO_edge2"].array(library="np"),axis=0)

        #Get jet kinematics
        jet_pts = np.append(jet_pts, tree["AntiKt10UFOCSSKSoftDropBeta100Zcut10JetsCalib_jetPt"].array(library="np"))
        jet_truth_pts = np.append(jet_truth_pts, tree["Truth_jetPt"].array(library="np"))
        jet_truth_etas = np.append(jet_truth_etas, ak.to_numpy(tree["Truth_jetEta"].array(library="np")))

        jet_truth_dRmatched = np.append(jet_truth_dRmatched, tree["Akt10TruthJet_dRmatched_particle_flavor"].array(library="np"))
    
        jet_truth_split = np.append(jet_truth_split, tree["AntiKt10UFOCSSKSoftDropBeta100Zcut10JetsCalib_Split12"].array(library="np"))
        jet_ms = np.append(jet_ms, ak.to_numpy(tree["AntiKt10UFOCSSKSoftDropBeta100Zcut10JetsCalib_jetM"].array()))
        jet_ungroomed_ms = np.append(jet_ungroomed_ms, tree["Akt10TruthJet_ungroomed_truthJet_m"].array(library="np"))
        jet_ungroomed_pts = np.append(jet_ungroomed_pts, tree["Akt10TruthJet_ungroomed_truthJet_pt"].array(library="np"))

        #Get Lund variables
        all_lund_zs = np.append(all_lund_zs,tree["AntiKt10UFOCSSKSoftDropBeta100Zcut10JetsCalib_jetLundz"].array(library="np") ) 
        all_lund_kts = np.append(all_lund_kts, tree["AntiKt10UFOCSSKSoftDropBeta100Zcut10JetsCalib_jetLundKt"].array(library="np") ) 
        all_lund_drs = np.append(all_lund_drs, tree["AntiKt10UFOCSSKSoftDropBeta100Zcut10JetsCalib_jetLundDeltaR"].array(library="np") )



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

batch_size = 2048

dataset= shuffle(dataset,random_state=42)
train_ds, validation_ds = train_test_split(dataset, test_size = 0.2, random_state = 144)
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


model = LundNet()
device = torch.device('cuda') # Usually gpu 4 worked best, it had the most memory available
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)


def train(loader):
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

@torch.no_grad()
def get_accuracy(loader):
    #remember to change this when evaluating combined model
    model.eval()
    correct = 0
    for data in loader:
        cl_data = data.to(device)
        new_y = torch.reshape(cl_data.y, (int(list(cl_data.y.shape)[0]),1))
        pred = model(cl_data).max(dim=1)[1]
        correct += pred.eq(new_y[0,:]).sum().item()
    return correct / len(loader.dataset)

@torch.no_grad()
def my_test (loader):
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
def get_scores(loader):
    model.eval()
    total_output = np.array([[1,1]])
    for data in loader:
        cl_data = data.to(device)
        pred = model(cl_data)
        total_output = np.append(total_output, pred.cpu().detach().numpy(), axis=0)
    return total_output[1:]
    
model_name = "LundNet_slabel_"
path = "/sps/atlas/k/khandoga/TrainGNN/Models/"
train_loss = []
val_loss = []
train_acc = []
val_acc = []
n_epochs = 35
save_every_epoch = False
metrics_filename = path+"losses_"+model_name+datetime.now().strftime("%d%m-%H%M")+".txt"

for epoch in range(n_epochs):
    print("Epoch:{}".format(epoch+1))
    train_loss.append(train(train_loader))
#    print ("Train Loss:",train_loss[-1])
#    delta_t_fileax = time.time() - t_start
#    print("trained epoch in {:.4f} seconds.".format(delta_t_fileax))
    val_loss.append(my_test(val_loader))
#    print ("Val Loss:",val_loss[-1])
#    delta_t_fileax = time.time() - t_start
#    print("tested val loss in {:.4f} seconds.".format(delta_t_fileax))
    train_acc.append(get_accuracy(train_loader))
    val_acc.append(get_accuracy(val_loader))
    print('Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f}, Train Acc: {:.5f}, Val Acc: {:.5f}'.format(epoch+1, train_loss[epoch], val_loss[epoch], train_acc[epoch], val_acc[epoch]))
    metrics = pd.DataFrame({"Train_Loss":train_loss,"Val_Loss":val_loss,"Train_Acc":train_acc,"Val_Acc":val_acc})
    metrics.to_csv(metrics_filename, index = False)

#    print('Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(epoch+1, train_loss[epoch], val_loss[epoch]))
    if (save_every_epoch):
        torch.save(model.state_dict(), path+model_name+"e{:03d}".format(epoch+1)+"_{:.5f}".format(val_loss[epoch])+".pt")
    elif epoch == n_epochs-1:
        torch.save(model.state_dict(), path+model_name+"e{:03d}".format(epoch+1)+"_{:.5f}".format(val_loss[epoch])+".pt")


